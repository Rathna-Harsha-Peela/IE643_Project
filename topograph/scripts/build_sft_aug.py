"""sft_train_aug.jsonl = sft_train.jsonl (official train_ft) + rendered synthetic modules (same prompt/serialisation).

    python scripts/build_sft_aug.py --synthetic data/processed/synthetic.jsonl --tokenizer $MODELS/Qwen3.5-9B
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.io import PROC_DIR, read_jsonl, write_jsonl  # noqa: E402
from tgraph.prompts import build_messages, render  # noqa: E402
from tgraph.schema import canonical_to_cg, dumps_cg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=os.path.join(PROC_DIR, "synthetic.jsonl"))
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    audit = json.load(open(os.path.join(PROC_DIR, "token_audit.json")))
    base = read_jsonl(os.path.join(PROC_DIR, "sft_train.jsonl"))
    syn, dropped, lens = [], 0, []
    for r in read_jsonl(args.synthetic):
        prompt = render(tok, build_messages(r["source_code"]))
        comp = dumps_cg(canonical_to_cg(r["target_graph"])) + audit["turn_end"]
        n = len(tok(prompt + comp, add_special_tokens=False)["input_ids"])
        if n > audit["max_length"]:
            dropped += 1
            continue
        lens.append(n)
        syn.append({"prompt": prompt, "completion": comp, "sample_id": r["sample_id"], "tier": "SYN"})
    write_jsonl(os.path.join(PROC_DIR, "sft_train_aug.jsonl"), base + syn)
    lens.sort()
    print(f"official {len(base)} + synthetic {len(syn)} (dropped {dropped} over max_length {audit['max_length']}); "
          f"synthetic tokens max {lens[-1]} median {lens[len(lens) // 2]}")


if __name__ == "__main__":
    main()
