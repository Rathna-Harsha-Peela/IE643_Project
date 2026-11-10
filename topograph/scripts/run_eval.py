"""Per-sample validity + metrics for prediction files (spec §13). CPU only, multiprocessing.

    python scripts/run_eval.py --pred results/raw/<config>/<set>.jsonl [...]   or   --all
Writes <set>.metrics.jsonl next to each prediction file.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.failures import classify  # noqa: E402
from tgraph.io import REPO, load_eval_set, read_jsonl, write_jsonl  # noqa: E402
from tgraph.metrics import compute_metrics  # noqa: E402
from tgraph.schema import cg_to_canonical  # noqa: E402
from tgraph.validate import validate  # noqa: E402

META = ["sample_id", "tier", "augmentation_type", "size_bin", "struct_seen", "category"]


def evaluate_row(args):
    pred_row, ref_row = args
    cg = pred_row.get("cg")
    rep = validate(cg, ref_row["source_code"], input_shape=ref_row.get("input_shape_override"),
                   parse_info=pred_row.get("parse_info"), class_name=ref_row.get("class_name")).to_dict()
    pred = cg_to_canonical(cg) if cg else None
    m = compute_metrics(pred, ref_row["target_graph"])
    lv = rep["levels_passed"]
    n_mod = sum(1 for n in (cg or {}).get("nodes", []) if n.get("op") == "call_module")
    out = {k: ref_row.get(k) for k in META}
    out.update(config=pred_row.get("config"), parse_ok=float(bool((pred_row.get("parse_info") or {}).get("parse_ok"))),
               schema_ok=float(bool((pred_row.get("parse_info") or {}).get("schema_ok"))),
               repaired=float(bool((pred_row.get("parse_info") or {}).get("repaired"))),
               struct_valid=float(bool(lv.get("V1"))), arity_valid=float(bool(lv.get("V2"))),
               code_consistent=float(bool(lv.get("V3"))), shape_consistent=float(bool(lv.get("V4"))),
               fully_valid=float(rep["fully_valid"]), validity_score=rep["score"],
               n_call_module=n_mod, n_undeclared=rep["n_errors_by_code"].get("UNDECLARED_MODULE", 0),
               dead_code_included=float("DEAD_CODE_INCLUDED" in rep["n_errors_by_code"]),
               error_codes=sorted(rep["n_errors_by_code"]), rounds_used=pred_row.get("rounds_used", 0),
               initial_fully_valid=float(bool(pred_row.get("initial_fully_valid", rep["fully_valid"]))),
               n_output_tokens=pred_row.get("n_output_tokens"), finish_reason=pred_row.get("finish_reason"),
               latency_s=pred_row.get("latency_s"))
    out.update({k: v for k, v in m.items() if k not in ("pred_props", "ref_props")})
    out["failures"] = classify(pred, ref_row["target_graph"], m, rep)
    return out


def eval_file(path, procs):
    eval_set = os.path.basename(path).split(".")[0]
    refs = {r["sample_id"]: r for r in load_eval_set(eval_set)}
    preds = read_jsonl(path)
    jobs = [(p, refs[p["sample_id"]]) for p in preds if refs[p["sample_id"]].get("target_graph")]
    with Pool(procs) as pool:
        rows = pool.map(evaluate_row, jobs, chunksize=8)
    for r in rows:
        r["eval_set"] = eval_set
    out = path[: -len(".jsonl")] + ".metrics.jsonl"
    write_jsonl(out, rows)
    fv = sum(r["fully_valid"] for r in rows) / max(1, len(rows))
    ef = sum(r["edge_f1"] for r in rows) / max(1, len(rows))
    ex = sum(r["exact_strict"] for r in rows) / max(1, len(rows))
    print(f"{out}: n={len(rows)} fully_valid={fv:.3f} edge_f1={ef:.3f} exact={ex:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    files = list(args.pred)
    if args.all:
        files += [f for f in glob.glob(os.path.join(REPO, "results", "raw", "*", "*.jsonl"))
                  if not f.endswith(".metrics.jsonl") and ".limit" not in f
                  and os.path.basename(f) != "untraceable.jsonl"]   # no reference graph exists
    for f in sorted(set(files)):
        out = f[: -len(".jsonl")] + ".metrics.jsonl"
        if os.path.exists(out) and not args.force and os.path.getmtime(out) > os.path.getmtime(f):
            continue
        eval_file(f, args.procs)


if __name__ == "__main__":
    main()
