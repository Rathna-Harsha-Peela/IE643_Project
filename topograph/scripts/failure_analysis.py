"""Failure taxonomy summary (spec §9.3): counts per config x tier and 2 example ids per label.

    python scripts/failure_analysis.py
Writes results/tables/failure_counts.csv, failure_examples.json, failures.md.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.failures import LABELS  # noqa: E402
from tgraph.io import REPO, read_jsonl  # noqa: E402

TAB = os.path.join(REPO, "results", "tables")


def main():
    rows = []
    for f in glob.glob(os.path.join(REPO, "results", "raw", "*", "*.metrics.jsonl")):
        rows += read_jsonl(f)
    if not rows:
        print("no metrics yet")
        return
    df = pd.DataFrame(rows)
    ex = df.explode("failures")
    ex = ex[ex.failures.notna()]
    counts = ex.groupby(["config", "eval_set", "tier", "failures"]).size().rename("count").reset_index()
    counts.to_csv(os.path.join(TAB, "failure_counts.csv"), index=False)
    examples = {}
    for (cfg, es), g in ex.groupby(["config", "eval_set"]):
        examples.setdefault(cfg, {})[es] = {l: g[g.failures == l].sample_id.head(2).tolist() for l in LABELS
                                            if (g.failures == l).any()}
    with open(os.path.join(TAB, "failure_examples.json"), "w") as f:
        json.dump(examples, f, indent=1)
    piv = ex[ex.eval_set == "test"].groupby(["config", "failures"]).size().unstack(fill_value=0)
    n = df[df.eval_set == "test"].groupby("config").size()
    md = ["# Failure taxonomy (official test)\n", "Labels per 100 test samples (a sample can carry several).\n",
          "| config | " + " | ".join(l for l in LABELS if l in piv.columns) + " |",
          "|---|" + "---|" * sum(l in piv.columns for l in LABELS)]
    for c, r in piv.iterrows():
        md.append(f"| {c} | " + " | ".join(f"{100 * r[l] / n[c]:.1f}" for l in LABELS if l in piv.columns) + " |")
    with open(os.path.join(TAB, "failures.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
