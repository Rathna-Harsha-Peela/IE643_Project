"""Qualitative gallery (spec §10): per tier best/median/worst on test, OOD samples, untraceable examples.

    python scripts/gallery.py --config ft9b_c
Writes results/figures/gallery/*.svg and gallery.md.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph import visualize as V  # noqa: E402
from tgraph.io import PROC_DIR, REPO, load_eval_set, read_jsonl  # noqa: E402
from tgraph.schema import cg_to_canonical  # noqa: E402

GAL = os.path.join(REPO, "results", "figures", "gallery")


def entry(md, cfg, row, ref, met, tag):
    sid = row["sample_id"]
    base = os.path.join(GAL, f"{cfg}_{sid}")
    md.append(f"### {tag}: `{sid}`" + (f" (graph similarity {met['graph_similarity']:.2f}, "
                                        f"edge F1 {met['edge_f1']:.2f}, failures: {', '.join(met['failures']) or 'none'})"
                                        if met else ""))
    md.append(f"<details><summary>source</summary>\n\n```python\n{ref['source_code']}\n```\n</details>\n")
    pred = cg_to_canonical(row["cg"]) if row.get("cg") else None
    if ref.get("target_graph"):
        V.render(ref["target_graph"], base + "_ref", title="Reference (torch.fx)")
        md.append(f"![ref]({os.path.basename(base)}_ref.svg)")
    if pred:
        V.render(pred, base + "_pred", title=f"Predicted ({cfg})")
        md.append(f"![pred]({os.path.basename(base)}_pred.svg)")
        if ref.get("target_graph"):
            V.render_diff(pred, ref["target_graph"], base + "_diff")
            md.append(f"![diff]({os.path.basename(base)}_diff.svg)")
    else:
        md.append("_prediction could not be parsed_")
    md.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ft9b_c")
    args = ap.parse_args()
    os.makedirs(GAL, exist_ok=True)
    raw = os.path.join(REPO, "results", "raw", args.config)
    md = [f"# Gallery — {args.config}\n", "Reference / prediction / diff (red dashed = missing, orange = hallucinated, "
          "red text = wrong kind or shape).\n"]
    if os.path.exists(os.path.join(raw, "test.metrics.jsonl")):
        preds = {r["sample_id"]: r for r in read_jsonl(os.path.join(raw, "test.jsonl"))}
        mets = read_jsonl(os.path.join(raw, "test.metrics.jsonl"))
        refs = {r["sample_id"]: r for r in load_eval_set("test")}
        for t in ["L0", "L1", "L2", "L3", "L4"]:
            ms = sorted([m for m in mets if m["tier"] == t], key=lambda m: (m["graph_similarity"], m["sample_id"]))
            if not ms:
                continue
            md.append(f"## Tier {t}\n")
            for tag, m in (("best", ms[-1]), ("median", ms[len(ms) // 2]), ("worst", ms[0])):
                entry(md, args.config, preds[m["sample_id"]], refs[m["sample_id"]], m, f"{t} {tag}")
    if os.path.exists(os.path.join(raw, "ood.metrics.jsonl")):
        md.append("## Out-of-distribution modules\n")
        preds = {r["sample_id"]: r for r in read_jsonl(os.path.join(raw, "ood.jsonl"))}
        mets = read_jsonl(os.path.join(raw, "ood.metrics.jsonl"))
        refs = {r["sample_id"]: r for r in load_eval_set("ood")}
        seen = set()
        picks = []
        for m in sorted(mets, key=lambda m: m["sample_id"]):
            if m["category"] not in seen:
                seen.add(m["category"])
                picks.append(m)
        for m in picks[:6]:
            entry(md, args.config, preds[m["sample_id"]], refs[m["sample_id"]], m, f"OOD {m['category']}")
    up = os.path.join(raw, "untraceable.jsonl")
    if os.path.exists(up) and os.path.exists(os.path.join(PROC_DIR, "untraceable.jsonl")):
        md.append("## Untraceable code (no torch.fx reference exists)\n")
        refs = {r["sample_id"]: r for r in load_eval_set("untraceable")}
        for r in read_jsonl(up)[:3]:
            entry(md, args.config, r, refs[r["sample_id"]], None, "untraceable")
    with open(os.path.join(GAL, "gallery.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", os.path.join(GAL, "gallery.md"))


if __name__ == "__main__":
    main()
