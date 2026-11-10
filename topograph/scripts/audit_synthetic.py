"""Leakage audit: trained synthetic modules vs novel_topo/OOD (structure overlap, closest source-token Jaccard).

    python scripts/audit_synthetic.py
"""
import json, re, sys
sys.path.insert(0, ".")
from scripts.prepare_data import struct_ids
from tgraph.io import load_eval_set, read_jsonl
if __name__ == "__main__":
    syn = read_jsonl("data/processed/synthetic.jsonl")
    sft_ids = {json.loads(l)["sample_id"] for l in open("data/processed/sft_train_aug.jsonl")}
    syn = [s for s in syn if s["sample_id"] in sft_ids]           # the ones actually trained on
    S, SK = {s["struct_id"] for s in syn}, {s["struct_id_kind"] for s in syn}
    tok = lambda src: set(re.findall(r"[A-Za-z_]+", src))
    syn_toks = [(s["sample_id"], tok(s["source_code"])) for s in syn]
    for name in ["ood", "novel_topo"]:
        ev = load_eval_set(name)
        es, ek = [struct_ids(r["target_graph"]) for r in ev], None
        print(f"{name}: {len(ev)} samples; unlabelled-structure overlap with trained synthetic: "
              f"{sum(a in S for a, _ in es)}; kind-labelled overlap: {sum(b in SK for _, b in es)}")
    ood = load_eval_set("ood")
    best = []
    for r in ood:
        t = tok(r["source_code"])
        j, sid = max((len(t & u) / len(t | u), sid) for sid, u in syn_toks)
        best.append((round(j, 2), r["sample_id"], sid))
    best.sort(reverse=True)
    print("highest source-token Jaccard OOD vs synthetic:", best[:5], " median:", best[len(best) // 2][0])
    sizes = sorted(len(s["target_graph"]["nodes"]) for s in syn)
    print("synthetic trained:", len(syn), "structures:", len(S))
