"""RL (PPO / GRPO Clipped Policy Optimization) for TopoGraph using Validator + Oracle Metrics Reward.

Usage (starting from Base Model or existing SFT LoRA adapter):
    python scripts/train_ppo.py \
        --base_model /kaggle/working/models/qwen35-9b-topograph-merged \
        --output_dir models/qwen35-9b-topograph-ppo-lora \
        --max_steps 200
"""
import argparse
import json
import os
import sys
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_sft import lora_targets
from tgraph.io import PROC_DIR, REPO, read_jsonl
from tgraph.metrics import compute_metrics
from tgraph.parse import parse_output
from tgraph.prompts import build_messages, render
from tgraph.schema import cg_to_canonical
from tgraph.validate import validate


def load_qlora_model(path, compute_dtype=torch.float16):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa"
        )
    except Exception:
        m = AutoModelForImageTextToText.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa"
        )
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
    return m


def compute_topograph_reward(completion_text: str, record: dict) -> float:
    """Composite reward: [-1.0, +2.0] based on syntax, static validator, and ground-truth graph metrics."""
    cg, pinfo = parse_output(completion_text)
    if cg is None:
        return -1.0  # Heavy penalty for unparseable JSON

    rep = validate(
        cg,
        record["source_code"],
        input_shape=record.get("input_shape"),
        parse_info=pinfo,
        class_name=record.get("class_name"),
    )

    # 1. Validator reward component (0.0 to 0.6)
    r_val = 0.4 * float(rep.score) + (0.2 if rep.fully_valid else 0.0)
    errs = rep.to_dict().get("n_errors_by_code", {})
    if "UNDECLARED_MODULE" in errs:
        r_val -= 0.25
    if "DEAD_CODE_INCLUDED" in errs:
        r_val -= 0.25

    # 2. Ground-truth Oracle graph matching component (0.0 to 1.4)
    r_gt = 0.0
    if record.get("target_graph"):
        try:
            pred_canon = cg_to_canonical(cg)
            m = compute_metrics(pred_canon, record["target_graph"])
            r_gt = (
                0.45 * float(m["edge_f1"])
                + 0.35 * float(m["graph_similarity"])
                + 0.20 * float(m["shape_acc"])
                + 0.40 * float(m["exact_strict"])
            )
        except Exception:
            r_gt = -0.3

    return float(r_val + r_gt)


def build_rl_dataset(tok, split_ids_file="splits.json", records_file="records.jsonl", limit=None):
    splits = json.load(open(os.path.join(PROC_DIR, split_ids_file)))
    train_ids = set(splits["train_ft"])
    recs = [r for r in read_jsonl(os.path.join(PROC_DIR, records_file)) if r["sample_id"] in train_ids]
    if limit:
        recs = recs[:limit]

    rows = []
    lookup = {}
    for r in recs:
        prompt = render(tok, build_messages(r["source_code"]))
        rows.append({"prompt": prompt, "sample_id": r["sample_id"]})
        lookup[r["sample_id"]] = r
    return Dataset.from_list(rows), lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="Path to base Qwen3.5-9B or merged SFT model")
    ap.add_argument("--init_adapter", default=None, help="Optional existing LoRA adapter to continue from")
    ap.add_argument("--output_dir", default=os.path.join(REPO, "models", "qwen35-9b-topograph-ppo-lora"))
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_generations", type=int, default=2, help="Rollouts per prompt for advantage estimation")
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cc = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if cc >= 8 else torch.float16

    model = load_qlora_model(args.base_model, compute_dtype=dtype)
    print(f"Loaded QLoRA model from {args.base_model} in {dtype}")

    _, targets, regex = lora_targets(model)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=regex,
    )

    dataset, rec_lookup = build_rl_dataset(tok, limit=args.limit)

    def reward_fn(prompts, completions, sample_id=None, **kwargs):
        rewards = []
        for i, comp in enumerate(completions):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            sid = sample_id[i] if sample_id is not None else None
            rec = rec_lookup.get(sid)
            rewards.append(compute_topograph_reward(text, rec) if rec else -1.0)
        return rewards

    from trl import GRPOConfig, GRPOTrainer

    rl_cfg = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
                max_steps=args.max_steps,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_steps=50,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.04,
        report_to="none",
        seed=643,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=rl_cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=peft_cfg,
    )

    print("Starting PPO/GRPO policy optimization...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved RL fine-tuned LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
