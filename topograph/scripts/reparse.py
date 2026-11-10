"""Re-parse and re-validate stored raw outputs with the current parser (no generation).

    python scripts/reparse.py results/raw/<config>/<set>.jsonl [...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.generate import _check, load_config  # noqa: E402
from tgraph.io import load_eval_set, read_jsonl, write_jsonl  # noqa: E402


def main():
    for path in sys.argv[1:]:
        cfg = load_config(os.path.basename(os.path.dirname(path)))
        refs = {r["sample_id"]: r for r in load_eval_set(os.path.basename(path).split(".")[0])}
        rows = read_jsonl(path)
        for o in rows:
            cg, pinfo, rep = _check(o["raw_text"], refs[o["sample_id"]], bool(cfg.get("thinking")))
            o.update(cg=cg, parse_info=pinfo, validation=rep.to_dict())
            if not cfg.get("retry_rounds"):
                o.update(initial_cg=cg, initial_fully_valid=rep.fully_valid)
        write_jsonl(path, rows)
        print(path, "fully_valid", sum(o["validation"]["fully_valid"] for o in rows) / len(rows),
              "| no answer", sum(o["cg"] is None for o in rows))


if __name__ == "__main__":
    main()
