"""Store predictions for the app's examples in app/precomputed.jsonl so "Precomputed" mode works from a fresh clone
(results/raw is not shipped).   python scripts/build_app_cache.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import core  # noqa: E402
from tgraph.io import REPO, write_jsonl  # noqa: E402


def main():
    pre = core.precomputed()
    rows = []
    for label, code in core.examples():
        h = core.src_hash(code)
        for cfg, table in pre.items():
            r = table.get(h)
            if r:
                rows.append({"config": cfg, "hash": h, "example": label, "cg": r.get("cg"), "raw_text": r.get("raw_text"),
                             "parse_info": r.get("parse_info"), "rounds_used": r.get("rounds_used", 0)})
    write_jsonl(os.path.join(REPO, "app", "precomputed.jsonl"), rows)
    print(f"{len(rows)} stored predictions for {len(core.examples())} examples x {len(pre)} configs")


if __name__ == "__main__":
    main()
