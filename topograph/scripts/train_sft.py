"""LoRA SFT of Qwen3.5 on CG-JSON completions (spec §11). Env: tg-train.

    python scripts/train_sft.py --config configs/train_9b.yaml [--max_steps 20 --timing_only]
Resumable: re-running picks up the newest checkpoint in output_dir.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.io import PROC_DIR, REPO, read_jsonl  # noqa: E402

STD_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load_model(path, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
        how = f"AutoModelForCausalLM -> {type(m).__name__}"
    except Exception as e:  # noqa: BLE001
        log(f"AutoModelForCausalLM failed ({type(e).__name__}: {e}); trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
        how = f"AutoModelForImageTextToText -> {type(m).__name__}"
    return m, how


def lora_targets(model):
    """Leaf nn.Linear names in the language model (vision tower excluded) -> regex for PEFT target_modules."""
    leaves = sorted({n.split(".")[-1] for n, mod in model.named_modules()
                     if isinstance(mod, torch.nn.Linear) and "visual" not in n and "lm_head" not in n
                     and not n.startswith("mtp")})
    # Gated DeltaNet projections; in_proj_a / in_proj_b only output one value per head, too small for LoRA
    extra = [x for x in ("in_proj_qkv", "in_proj_z", "in_proj_qkvz", "out_proj") if x in leaves]
    targets = [t for t in STD_TARGETS if t in leaves] + extra
    regex = r"^(?!.*(visual|mtp)).*\.(" + "|".join(targets) + r")$"
    return leaves, targets, regex


def fast_path_report():
    """transformers uses fla's kernels for Gated DeltaNet iff `import fla` succeeds (decided at import time)."""
    rep = {"cuda": torch.cuda.is_available()}
    try:
        import fla  # noqa: F401
        rep["fla"] = getattr(fla, "__version__", "yes")
        rep["gated_deltanet_kernel"] = "fla (fast)" if torch.cuda.is_available() else "fla (no GPU!)"
    except Exception as e:  # noqa: BLE001
        rep["fla"] = f"missing ({type(e).__name__})"
    try:
        import causal_conv1d  # noqa: F401
        rep["causal_conv1d"] = "yes"
    except Exception:  # noqa: BLE001
        rep["causal_conv1d"] = "missing"
    rep.setdefault("gated_deltanet_kernel", "torch reference (slow)")
    return rep


def last_checkpoint(out):
    cks = sorted(glob.glob(os.path.join(out, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    return cks[-1] if cks else None


def sanity_generate(model, tok, n=20, max_new=1536):
    """Greedy-decode n val samples with the trained model; log exact-match vs gold CG."""
    from tgraph.parse import parse_output
    from tgraph.schema import canonical_to_cg
    rows = read_jsonl(os.path.join(PROC_DIR, "sft_val.jsonl"))
    val = {r["sample_id"]: r for r in read_jsonl(os.path.join(PROC_DIR, "val.jsonl"))}
    rows = rows[:: max(1, len(rows) // n)][:n]
    model.eval()
    tok.padding_side = "left"
    hits = 0
    for i in range(0, len(rows), 4):
        chunk = rows[i:i + 4]
        enc = tok([r["prompt"] for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, r in enumerate(chunk):
            text = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            cg, _ = parse_output(text)
            hits += int(cg == canonical_to_cg(val[r["sample_id"]]["target_graph"]))
    return hits, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--timing_only", action="store_true", help="no saving/eval; report s/step and peak memory")
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--no_sanity", action="store_true")
    ap.add_argument("--max_length", type=int, default=None, help="smoke tests only: override the audited max_length")
    args = ap.parse_args()
    C = yaml.safe_load(open(args.config))
    C["model_dir"] = os.path.expandvars(C["model_dir"])
    out = os.path.join(REPO, C["output_dir"])
    os.makedirs(out, exist_ok=True)
    from tgraph.compat import cpu_disable_fla
    if cpu_disable_fla():
        log("CPU run: fla disabled, Gated DeltaNet uses the reference PyTorch path")
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    import transformers
    import trl
    import peft
    log(f"torch {torch.__version__} transformers {transformers.__version__} trl {trl.__version__} peft {peft.__version__}")
    if torch.cuda.is_available():
        log("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    log("fast path:", fast_path_report())

    tok = AutoTokenizer.from_pretrained(C["model_dir"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, how = load_model(C["model_dir"])
    log("loaded with", how)
    leaves, targets, regex = lora_targets(model)
    log("Linear leaf names (language model):", leaves)
    log("LoRA targets:", targets)
    model.config.use_cache = False

    data = {"train": os.path.join(PROC_DIR, C.get("train_file", "sft_train.jsonl")),
            "val": os.path.join(PROC_DIR, C.get("val_file", "sft_val.jsonl"))}
    ds = load_dataset("json", data_files=data)
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ("prompt", "completion")])
    if args.limit_rows:                      # smoke tests: the shortest rows, so short max_length truncates nothing
        for split in ("train", "val"):
            lens = [len(p) + len(c) for p, c in zip(ds[split]["prompt"], ds[split]["completion"])]
            keep = sorted(range(len(lens)), key=lambda i: lens[i])[: args.limit_rows]
            ds[split] = ds[split].select(sorted(keep))
    if C.get("aug_subset"):                 # timing rule fallback: keep only some augmentation variants
        keep = set(C["aug_subset"])
        ids = {r["sample_id"]: r for r in read_jsonl(os.path.join(PROC_DIR, "records.jsonl"))}
        raw = read_jsonl(data["train"])
        idx = [i for i, r in enumerate(raw) if ids[r["sample_id"]]["augmentation_type"] in keep]
        ds["train"] = ds["train"].select(idx)
    audit = json.load(open(os.path.join(PROC_DIR, "token_audit.json")))
    timing = args.timing_only
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", os.path.join(out, "tb"))
    # transformers v5: warmup_steps < 1 is a ratio; TRL appends eos_token to completions that lack it, so it is set
    # to the chat template's turn-end token (already at the end of every completion) -> exactly one turn-end token
    sc = dict(output_dir=out, max_length=audit["max_length"], num_train_epochs=C["epochs"],
              learning_rate=float(C["lr"]), per_device_train_batch_size=C["batch_size"],
              per_device_eval_batch_size=C["batch_size"], gradient_accumulation_steps=C["grad_accum"],
              lr_scheduler_type="cosine", warmup_steps=0.03, bf16=torch.cuda.is_available(),
              gradient_checkpointing=True, logging_steps=1 if timing else 10,
              eval_strategy="no" if timing else "steps", eval_steps=C.get("eval_steps", 50),
              save_strategy="no" if timing else "steps", save_steps=C.get("save_steps", 50), save_total_limit=2,
              report_to="none" if timing else "tensorboard", seed=643, eos_token=audit["turn_end"],
              packing=False, completion_only_loss=None, dataloader_num_workers=2,
              gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.max_steps:
        sc["max_steps"] = args.max_steps
    if args.max_length:
        sc["max_length"] = args.max_length
    if not torch.cuda.is_available():
        sc.update(use_cpu=True, bf16=False, dataloader_num_workers=0)
    cfg = SFTConfig(**sc)
    lcfg = LoraConfig(r=C["lora_r"], lora_alpha=C["lora_alpha"], lora_dropout=C["lora_dropout"],
                      task_type="CAUSAL_LM", target_modules=regex)

    class Timer(TrainerCallback):
        def __init__(self):
            self.t = []

        def on_step_begin(self, a, s, c, **kw):
            self.t0 = time.time()

        def on_step_end(self, a, s, c, **kw):
            self.t.append(time.time() - self.t0)

    timer = Timer()
    trainer = SFTTrainer(model=model, processing_class=tok, args=cfg, train_dataset=ds["train"],
                         eval_dataset=None if timing else ds["val"], peft_config=lcfg, callbacks=[timer])
    tm = trainer.model
    n_tr = sum(p.numel() for p in tm.parameters() if p.requires_grad)
    n_vis = sum(p.numel() for n, p in tm.named_parameters() if p.requires_grad and "visual" in n)
    log(f"trainable params {n_tr / 1e6:.1f}M (vision tower trainable: {n_vis})")
    assert n_vis == 0

    # show one example exactly as the trainer sees it: the collator's labels (-100 = not trained)
    ex = trainer.train_dataset[0]
    log("dataset fields:", sorted(ex.keys()))
    batch = trainer.data_collator([ex])
    ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    trained = [i for i, l in enumerate(labels) if l != -100]
    if not trained:
        raise SystemExit(f"no trained tokens in example 0 ({len(ids)} tokens): max_length truncated the completion")
    first = trained[0]
    turn_end = tok.convert_tokens_to_ids(audit["turn_end"])
    log("PROMPT END:", repr(tok.decode(ids[max(0, first - 12):first])))
    log("TRAINED START:", repr(tok.decode(ids[first:first + 30])), "| TRAINED END:", repr(tok.decode(ids[trained[-1] - 5:trained[-1] + 1])))
    log(f"tokens {len(ids)}, trained {len(trained)} (contiguous: {trained == list(range(first, trained[-1] + 1))}); "
        f"turn-end tokens trained: {sum(1 for i in trained if ids[i] == turn_end)}; last trained = {tok.convert_ids_to_tokens(ids[trained[-1]])!r}")
    ptoks = tok(ds["train"][0]["prompt"], add_special_tokens=False)["input_ids"]
    log(f"prompt tokens identical to inference rendering: {ids[:first] == ptoks}")
    steps_per_epoch = len(trainer.get_train_dataloader()) // C["grad_accum"]
    log(f"train rows {len(ds['train'])}, optimizer steps/epoch ~{steps_per_epoch}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ck = None if timing else last_checkpoint(out)
    log("resume from", ck)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=ck)
    wall = time.time() - t0
    s_step = sorted(timer.t[2:])[len(timer.t[2:]) // 2] if len(timer.t) > 3 else (timer.t[-1] if timer.t else float("nan"))
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    res = {"config": args.config, "s_per_step_median": s_step, "peak_mem_gib": peak, "wall_s": wall,
           "steps": len(timer.t), "steps_per_epoch": steps_per_epoch,
           "eta_1_epoch_h": s_step * steps_per_epoch / 3600, "eta_2_epoch_h": 2 * s_step * steps_per_epoch / 3600,
           "load": how, "targets": targets, "fast_path": fast_path_report()}
    log("TIMING", json.dumps(res))
    if timing:
        with open(os.path.join(out, "timing.json"), "w") as f:
            json.dump(res, f, indent=1)
        return
    adapter_dir = os.path.join(REPO, "models", C["tag"] + "-lora")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    with open(os.path.join(adapter_dir, "training_args.json"), "w") as f:
        json.dump({**C, **{k: str(v) for k, v in sc.items()}, "result": res}, f, indent=1)
    hist = trainer.state.log_history
    keys = sorted({k for h in hist for k in h})
    with open(os.path.join(out, "log_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(hist)
    log("saved adapter to", adapter_dir)
    if not args.no_sanity:
        model = trainer.model
        model.config.use_cache = True
        hits, n = sanity_generate(model, tok, n=C.get("sanity_n", 20), max_new=C.get("sanity_max_new", 1536))
        log(f"SANITY exact-match on {n} val samples: {hits}/{n}")
        if hits / max(1, n) < 0.5:
            log("WARNING: sanity exact-match < 50%: check chat template, completion mask, turn-end token")
        with open(os.path.join(adapter_dir, "sanity.json"), "w") as f:
            json.dump({"exact": hits, "n": n}, f)


if __name__ == "__main__":
    main()
