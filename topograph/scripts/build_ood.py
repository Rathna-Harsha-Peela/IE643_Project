"""Label the hand-written OOD modules with the oracle (spec §7.2) and write data/processed/{ood,untraceable}.jsonl.

    python scripts/build_ood.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.io import PROC_DIR, REPO, write_jsonl  # noqa: E402
from tgraph.oracle import extract_reference_graph  # noqa: E402
from tgraph.schema import canonical_to_cg  # noqa: E402
from tgraph.validate import validate  # noqa: E402

OOD = os.path.join(REPO, "data", sys.argv[1] if len(sys.argv) > 1 else "ood")
NAME = sys.argv[1] if len(sys.argv) > 1 else "ood"


def size_bin(n):
    return "<=10" if n <= 10 else "11-15" if n <= 15 else "16-20" if n <= 20 else ">20"


def main():
    meta = json.load(open(os.path.join(OOD, "meta.json")))
    rows, dropped = [], []
    for name, m in sorted(meta.items()):
        src = open(os.path.join(OOD, f"{name}.py")).read()
        shp = m["input_shape"]
        multi = isinstance(shp[0], list)
        try:
            g = extract_reference_graph(src, input_shape=[tuple(s) for s in shp] if multi else tuple(shp),
                                        class_name=m["class_name"])
        except Exception as e:  # noqa: BLE001
            dropped.append((name, f"{type(e).__name__}: {e}"))
            continue
        if not multi:   # the stated shape must equal what the prompt's rule 7 implies
            try:
                implied = list(extract_reference_graph(src, class_name=m["class_name"])["nodes"][0]["shape"])
            except Exception as e:  # noqa: BLE001
                implied = f"n/a ({type(e).__name__})"
            assert implied == shp, (name, implied, shp)
        if any(n["shape"] is None for n in g["nodes"]):
            dropped.append((name, "non-tensor node (shape None)"))
            continue
        rep = validate(canonical_to_cg(g), src, class_name=m["class_name"])
        rows.append({"sample_id": f"{NAME.upper()}_{name}", "tier": NAME.upper(), "category": m["category"], "augmentation_type": "none",
                     "size_bin": size_bin(len(g["nodes"])), "struct_seen": False, "source_code": src,
                     "target_graph": g, "input_shape": shp, "class_name": m["class_name"],
                     "gold_validator_errors": [e["msg"] for e in rep.errors]})
        flag = "" if not rep.errors else f"  gold validator errors: {[e['msg'] for e in rep.errors][:3]}"
        print(f"{name:22s} {m['category']:18s} nodes={len(g['nodes']):3d} edges={len(g['edges']):3d}{flag}")
    write_jsonl(os.path.join(PROC_DIR, f"{NAME}.jsonl"), rows)
    print(f"{NAME}.jsonl: {len(rows)} samples; dropped {dropped}")
    if not os.path.exists(os.path.join(OOD, "untraceable", "meta.json")):
        return
    um = json.load(open(os.path.join(OOD, "untraceable", "meta.json")))
    urows = []
    for name, m in sorted(um.items()):
        src = open(os.path.join(OOD, "untraceable", f"{name}.py")).read()
        try:
            extract_reference_graph(src, input_shape=(1, 3, 16, 16), class_name=m["class_name"])
            status = "TRACEABLE?!"
        except Exception as e:  # noqa: BLE001
            status = f"{type(e).__name__}"
        print(f"untraceable {name:28s} oracle -> {status}")
        urows.append({"sample_id": f"UNT_{name}", "tier": "UNTRACEABLE", "category": "untraceable",
                      "augmentation_type": "none", "size_bin": None, "struct_seen": False, "source_code": src,
                      "target_graph": None, "input_shape": None, "class_name": m["class_name"], "oracle_error": status})
    write_jsonl(os.path.join(PROC_DIR, "untraceable.jsonl"), urows)


if __name__ == "__main__":
    main()
