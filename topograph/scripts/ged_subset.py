"""True GED vs AED on a stratified 100-sample subset of test per config (spec §9.2, figure F8).

    python scripts/ged_subset.py
Writes results/tables/ged_subset.csv (config, sample_id, aed, ged, ged_exact).
"""
import glob
import os
import random
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.io import REPO, load_eval_set, read_jsonl  # noqa: E402
from tgraph.metrics import compute_metrics, true_ged  # noqa: E402
from tgraph.schema import cg_to_canonical  # noqa: E402


def one(args):
    cfg, sid, cg, ref = args
    pred = cg_to_canonical(cg)
    ged, exact = true_ged(pred, ref, timeout=2.0)
    return {"config": cfg, "sample_id": sid, "aed": compute_metrics(pred, ref)["aed"], "ged": ged, "ged_exact": exact}


def main():
    refs = {r["sample_id"]: r for r in load_eval_set("test")}
    rng = random.Random(643)
    by_tier = {}
    for sid, r in sorted(refs.items()):
        by_tier.setdefault(r["tier"], []).append(sid)
    subset = [s for t in sorted(by_tier) for s in rng.sample(by_tier[t], 20)]
    jobs = []
    for f in sorted(glob.glob(os.path.join(REPO, "results", "raw", "*", "test.jsonl"))):
        cfg = os.path.basename(os.path.dirname(f))
        preds = {r["sample_id"]: r for r in read_jsonl(f)}
        jobs += [(cfg, s, preds[s]["cg"], refs[s]["target_graph"]) for s in subset if preds.get(s, {}).get("cg")]
    with Pool(min(16, os.cpu_count() or 4)) as p:
        rows = p.map(one, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    out = os.path.join(REPO, "results", "tables", "ged_subset.csv")
    df.to_csv(out, index=False)
    if len(df):
        from scipy.stats import spearmanr
        print(f"{len(df)} pairs; exact GED for {df.ged_exact.mean():.0%}; AED>=GED in {(df.aed >= df.ged - 1e-9).mean():.0%}; "
              f"Spearman {spearmanr(df.aed, df.ged).correlation:.3f}")


if __name__ == "__main__":
    main()
