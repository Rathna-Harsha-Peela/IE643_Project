"""Inference (spec §12): vLLM (GPU, default) or HF transformers backend, optional constrained decoding,
verify-and-retry, self-consistency, thinking mode.  Resumable: finished (config, eval set) files are skipped.

    python scripts/generate.py --config ft9b_c --eval_sets test,novel_topo --backend vllm [--limit N]
Writes results/raw/<config>/<eval_set>.jsonl, one row per sample.
"""
import argparse
import json
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.io import PROC_DIR, REPO, load_eval_set, read_jsonl, write_jsonl  # noqa: E402
from tgraph.parse import parse_output  # noqa: E402
from tgraph.prompts import RETRY_TEMPLATE, build_messages, render  # noqa: E402
from tgraph.schema import SCHEMA, canonical_to_cg, dumps_cg  # noqa: E402
from tgraph.validate import validate  # noqa: E402

RAW_OUT = os.path.join(REPO, "results", "raw")


def load_config(name, path=os.path.join(REPO, "configs", "experiments.yaml")):
    for c in yaml.safe_load(open(path)):
        if c["name"] == name:
            c = dict(c)
            c["model_dir"] = os.path.expandvars(c["model_dir"])
            if c.get("adapter"):
                c["adapter"] = os.path.expandvars(c["adapter"])
            return c
    raise KeyError(f"config {name} not in {path}")


def few_shot_exemplars():
    splits = json.load(open(os.path.join(PROC_DIR, "splits.json")))
    recs = {r["sample_id"]: r for r in read_jsonl(os.path.join(PROC_DIR, "records.jsonl"))}
    return [(recs[i]["source_code"], recs[i]["target_graph"]) for i in splits["few_shot_exemplars"]]


def shape_hint(row):
    """Only for multi-input OOD rows (rule 7 cannot fix the order otherwise) or explicit overrides."""
    s = row.get("input_shape_hint")
    return s if s else None


# ---------------------------------------------------------------------------------------------------------------
class VLLMBackend:
    def __init__(self, cfg):
        from vllm import LLM
        kw = dict(model=cfg["model_dir"], dtype=cfg.get("dtype", "half"),
                  tensor_parallel_size=cfg.get("tensor_parallel_size", 2),
                  disable_custom_all_reduce=True,
                  enforce_eager=cfg.get("enforce_eager", True),
                  max_model_len=cfg.get("max_model_len", 4096),
                  gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85), seed=643,
                  enable_prefix_caching=cfg.get("prefix_caching", False))   # experimental for Qwen3.5 linear-attn layers
        if cfg.get("adapter"):
            kw.update(enable_lora=True, max_lora_rank=cfg.get("lora_rank", 32))
        if cfg.get("max_num_seqs"):              # big models: fewer parallel sequences fit next to the weights
            kw["max_num_seqs"] = cfg["max_num_seqs"]
        if cfg.get("reasoning_parser"):          # grammar is enforced only after the reasoning (</think>) ends
            kw["reasoning_parser"] = cfg["reasoning_parser"]
        try:
            self.llm = LLM(limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
        except TypeError:
            self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.lora = None
        if cfg.get("adapter"):
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("ft", 1, cfg["adapter"])

    def params(self, cfg, constrained, n=1):
        from vllm import SamplingParams
        s = dict(cfg.get("sampling") or {})
        kw = dict(temperature=s.pop("temperature", 0.0), max_tokens=cfg.get("max_tokens", 3072), n=n, seed=643, **s)
        if constrained:
            try:
                from vllm.sampling_params import StructuredOutputsParams
                kw["structured_outputs"] = StructuredOutputsParams(json=SCHEMA)
            except ImportError:
                from vllm.sampling_params import GuidedDecodingParams
                kw["guided_decoding"] = GuidedDecodingParams(json=SCHEMA)
        return SamplingParams(**kw)

    def generate(self, prompts, cfg, constrained, n=1):
        t0 = time.time()
        outs = self.llm.generate(prompts, self.params(cfg, constrained, n), lora_request=self.lora)
        dt = (time.time() - t0) / max(1, len(prompts))
        res = []
        for o in outs:
            res.append([{"text": c.text, "n_output_tokens": len(c.token_ids), "finish_reason": c.finish_reason,
                         "n_prompt_tokens": len(o.prompt_token_ids), "latency_s": dt} for c in o.outputs])
        return res


class HFBackend:
    """Batched greedy generation with transformers (CPU or GPU). Constrained decoding is not applied here."""

    def __init__(self, cfg, batch_size=8):
        import torch
        from tgraph.compat import cpu_disable_fla
        cpu_disable_fla()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(cfg["model_dir"])
        self.tok.padding_side = "left"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(cfg["model_dir"], dtype=dtype).to(dev).eval()
        if cfg.get("adapter"):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, cfg["adapter"]).eval()
        self.dev, self.bs = dev, batch_size

    def generate(self, prompts, cfg, constrained, n=1):
        res = []
        for i in range(0, len(prompts), self.bs):
            chunk = prompts[i:i + self.bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            t0 = time.time()
            with self.torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=cfg.get("max_tokens", 3072), do_sample=False,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            dt = (time.time() - t0) / len(chunk)
            for j in range(len(chunk)):
                gen = out[j, enc["input_ids"].shape[1]:]
                ids = [t for t in gen.tolist() if t != (self.tok.pad_token_id or -1)]
                stopped = any(t in (self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")) for t in ids)
                res.append([{"text": self.tok.decode(ids, skip_special_tokens=True), "n_output_tokens": len(ids),
                             "finish_reason": "stop" if stopped else "length",
                             "n_prompt_tokens": int(enc["attention_mask"][j].sum()), "latency_s": dt}])
        return res


# ---------------------------------------------------------------------------------------------------------------
def _check(text, row, thinking=False):
    if thinking and "</think>" not in (text or ""):
        # reasoning never finished: any JSON in the text is a fragment of the reasoning, not an answer
        cg, pinfo = None, {"parse_ok": False, "repaired": False, "schema_ok": False, "coercions": 0,
                           "error": "no answer: reasoning did not finish (no </think>)"}
    else:
        cg, pinfo = parse_output(text)
    rep = validate(cg, row["source_code"], input_shape=row.get("input_shape_override"), parse_info=pinfo,
                   class_name=row.get("class_name"))
    return cg, pinfo, rep


def _rank(rep):
    return (rep.fully_valid, rep.score)


def run_set(backend, cfg, rows, exemplars):
    thinking = bool(cfg.get("thinking"))
    constrained = bool(cfg.get("constrained"))
    n_sc = int(cfg.get("self_consistency", 1))
    convs = [build_messages(r["source_code"], exemplars, shape_hint(r)) for r in rows]
    prompts = [render(backend.tok, m, thinking=thinking) for m in convs]
    gens = backend.generate(prompts, cfg, constrained, n=n_sc)
    out = []
    for r, m, p, g in zip(rows, convs, prompts, gens):
        cands = [(c, *_check(c["text"], r, thinking)) for c in g]
        if n_sc > 1:                                  # validator score, ties broken by edge-majority vote
            votes = {}
            for c, cg, _, _ in cands:
                for e in (cg or {}).get("edges", []):
                    votes[tuple(e)] = votes.get(tuple(e), 0) + 1
            best = max(cands, key=lambda t: (_rank(t[3]), sum(votes.get(tuple(e), 0) for e in (t[1] or {}).get("edges", []))))
        else:
            best = cands[0]
        c, cg, pinfo, rep = best
        out.append({"sample_id": r["sample_id"], "config": cfg["name"], "raw_text": c["text"],
                    "n_output_tokens": sum(x[0]["n_output_tokens"] for x in cands), "n_prompt_tokens": c["n_prompt_tokens"],
                    "finish_reason": c["finish_reason"], "latency_s": c["latency_s"], "cg": cg, "parse_info": pinfo,
                    "validation": rep.to_dict(), "rounds_used": 0, "initial_fully_valid": rep.fully_valid,
                    "initial_cg": cg, "_conv": m})
    # verify-and-retry: only validator feedback, never the reference
    for rnd in range(1, int(cfg.get("retry_rounds", 0)) + 1):
        todo = [o for o in out if o["validation"]["errors"]]
        if not todo:
            break
        rows_by = {r["sample_id"]: r for r in rows}
        convs2 = []
        for o in todo:
            errs = "\n".join(f"- {e['msg']}" for e in o["validation"]["errors"][:12])
            conv = o["_conv"] + [{"role": "assistant", "content": dumps_cg(o["cg"]) if o["cg"] else o["raw_text"]},
                                 {"role": "user", "content": RETRY_TEMPLATE.format(errors=errs)}]
            convs2.append(conv)
        prompts2 = [render(backend.tok, m, thinking=thinking) for m in convs2]
        gens2 = backend.generate(prompts2, cfg, True)
        for o, conv, g in zip(todo, convs2, gens2):
            c = g[0]
            cg, pinfo, rep = _check(c["text"], rows_by[o["sample_id"]], thinking)
            o["n_output_tokens"] += c["n_output_tokens"]
            o["rounds_used"] = rnd
            if _rank(rep) >= (o["validation"]["fully_valid"], o["validation"]["score"]):
                o.update(raw_text=c["text"], cg=cg, parse_info=pinfo, validation=rep.to_dict(),
                         finish_reason=c["finish_reason"], _conv=conv)
            else:
                o["_conv"] = conv
    for o in out:
        o.pop("_conv")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_sets", default=None, help="comma list; default = the config's eval_sets")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--hf_batch", type=int, default=8)
    ap.add_argument("--out_dir", default=RAW_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sets = args.eval_sets.split(",") if args.eval_sets else cfg["eval_sets"]
    todo = []
    for s in sets:
        path = os.path.join(args.out_dir, cfg["name"], f"{s}.jsonl")
        if os.path.exists(path) and not args.force and args.limit is None:
            print(f"[skip] {path} exists")
            continue
        if not os.path.exists(os.path.join(PROC_DIR, f"{s}.jsonl")):
            print(f"[skip] eval set {s} not built")
            continue
        todo.append((s, path))
    if not todo:
        return
    print(f"[{cfg['name']}] model={cfg['model_dir']} backend={args.backend} sets={[s for s, _ in todo]}", flush=True)
    backend = VLLMBackend(cfg) if args.backend == "vllm" else HFBackend(cfg, args.hf_batch)
    exemplars = few_shot_exemplars() if cfg.get("prompt_mode") == "few_shot" else None
    for s, path in todo:
        rows = load_eval_set(s)[: args.limit]
        t0 = time.time()
        out = run_set(backend, cfg, rows, exemplars)
        if args.limit is not None:
            path = path.replace(".jsonl", f".limit{args.limit}.jsonl")
        write_jsonl(path, out)
        fv = sum(o["validation"]["fully_valid"] for o in out) / len(out)
        print(f"[{cfg['name']}/{s}] n={len(out)} fully_valid={fv:.3f} wall={time.time() - t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
