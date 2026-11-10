"""Data preparation (spec §7): records, structure ids, novel-topology holdout, val split, eval sets, SFT files,
token audit, dataset stats.  Deterministic with seed 643.

    python scripts/prepare_data.py [--tokenizer $MODELS/Qwen3.5-9B] [--skip_oracle_check]
"""
import argparse
import collections
import json
import os
import random
import re
import sys

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgraph.io import PROC_DIR, REPO, load_raw_records, write_jsonl  # noqa: E402
from tgraph.schema import canonical_to_cg, dumps_cg, to_nx  # noqa: E402

SEED = 643
TIERS = ["L0", "L1", "L2", "L3", "L4"]
AUGS = ["none", "aug_variant_1", "aug_variant_2", "aug_variant_3"]
NOVEL_CAP = 120
EVAL_FIELDS = ["sample_id", "tier", "augmentation_type", "size_bin", "struct_seen", "struct_id", "source_code",
               "target_graph", "input_shape", "architecture_family_id"]


def size_bin(n):
    return "<=10" if n <= 10 else "11-15" if n <= 15 else "16-20" if n <= 20 else ">20"


def struct_ids(g):
    G = to_nx(g)
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from(G.edges)
    return (nx.weisfeiler_lehman_graph_hash(H, iterations=3),
            nx.weisfeiler_lehman_graph_hash(G, node_attr="kind", iterations=3))


def anonymize(src):
    m = re.search(r"class\s+(\w+)\s*\(\s*nn\.Module\s*\)", src)
    name = m.group(1)
    src = re.sub(rf"\bclass\s+{name}\b", "class Net", src)
    return re.sub(rf"super\(\s*{name}\s*,", "super(Net,", src)


def input_shape(g):
    phs = [n["shape"] for n in g["nodes"] if n["op"] == "placeholder"]
    return phs[0] if len(phs) == 1 else phs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=os.path.join(os.environ.get("MODELS", ""), "Qwen3.5-9B"))
    ap.add_argument("--extra_tokenizers", default="", help="comma list of other model dirs to check templates")
    ap.add_argument("--skip_oracle_check", action="store_true")
    args = ap.parse_args()
    os.makedirs(PROC_DIR, exist_ok=True)
    rng = random.Random(SEED)

    # 1-2. records + structure ids
    recs = load_raw_records()
    assert len(recs) == 5000, len(recs)
    for r in recs:
        g = r["target_graph"]
        r["n_nodes"], r["n_edges"] = len(g["nodes"]), len(g["edges"])
        r["size_bin"] = size_bin(r["n_nodes"])
        r["struct_id"], r["struct_id_kind"] = struct_ids(g)
        r["input_shape"] = input_shape(g)
    write_jsonl(os.path.join(PROC_DIR, "records.jsonl"), recs)
    by_id = {r["sample_id"]: r for r in recs}
    train = [r for r in recs if r["split"] == "train"]
    test = [r for r in recs if r["split"] == "test"]
    assert len(train) == 4000 and len(test) == 1000

    # 3. novel-topology holdout H (decided here, before any training)
    holdout, holdout_log = set(), {}
    for t in TIERS:
        cnt = collections.Counter(r["struct_id"] for r in train if r["tier"] == t)
        classes = sorted(cnt, key=lambda s: (-cnt[s], s))
        k = max(1, round(0.2 * len(classes)))
        pick = rng.sample(classes[1:], k)            # never the single largest class
        holdout |= set(pick)
        holdout_log[t] = {"n_classes": len(classes), "k": k,
                          "chosen": {s: cnt[s] for s in sorted(pick)},
                          "class_counts": [cnt[s] for s in classes]}
    novel = []
    for t in TIERS:
        fams = sorted({r["architecture_family_id"] for r in train if r["tier"] == t and r["struct_id"] in holdout})
        rng.shuffle(fams)
        keep, n = [], 0
        for f in fams:
            members = [r for r in train if r["architecture_family_id"] == f]
            if n + len(members) > NOVEL_CAP:
                break
            keep += members
            n += len(members)
        novel += sorted(keep, key=lambda r: r["sample_id"])
    ft_pool = [r for r in train if r["struct_id"] not in holdout]

    # 4. validation split: 5% of the remaining families per tier
    val, train_ft = [], []
    for t in TIERS:
        fams = sorted({r["architecture_family_id"] for r in ft_pool if r["tier"] == t})
        nval = max(1, round(0.05 * len(fams)))
        vf = set(rng.sample(fams, nval))
        val += [r for r in ft_pool if r["tier"] == t and r["architecture_family_id"] in vf]
        train_ft += [r for r in ft_pool if r["tier"] == t and r["architecture_family_id"] not in vf]

    # 5. official test, tagged seen/unseen structure
    for r in recs:
        r["struct_seen"] = r["struct_id"] not in holdout

    # few-shot exemplars: one L0, L2, L4 from train_ft, augmentation none
    exemplars = []
    for t in ["L0", "L2", "L4"]:
        cands = sorted(r["sample_id"] for r in train_ft if r["tier"] == t and r["augmentation_type"] == "none")
        exemplars.append(rng.choice(cands))

    # test_mini: 20 per tier, 5 per augmentation type
    mini = []
    for t in TIERS:
        for a in AUGS:
            c = sorted([r for r in test if r["tier"] == t and r["augmentation_type"] == a], key=lambda r: r["sample_id"])
            mini += rng.sample(c, 5)

    # 6. anonymized test
    anon = []
    for r in test:
        a = dict(r)
        a["source_code"] = anonymize(r["source_code"])
        anon.append(a)
    if not args.skip_oracle_check:
        from tgraph.oracle import extract_reference_graph
        for a in random.Random(SEED).sample(anon, 20):
            assert json.dumps(extract_reference_graph(a["source_code"]), sort_keys=True) == \
                json.dumps(a["target_graph"], sort_keys=True), a["sample_id"]
        print("anonymized test: oracle graph unchanged on 20 random samples")

    # 8. eval set files
    def ev(rows):
        return [{k: r.get(k) for k in EVAL_FIELDS} for r in rows]
    for name, rows in [("test", test), ("test_anon", anon), ("novel_topo", novel), ("test_mini", mini),
                       ("val", val)]:
        write_jsonl(os.path.join(PROC_DIR, f"{name}.jsonl"), ev(rows))

    # leakage guards
    ft_ids = {r["sample_id"] for r in train_ft}
    for name, rows in [("val", val), ("novel_topo", novel), ("test", test)]:
        assert not ft_ids & {r["sample_id"] for r in rows}, name
    assert not {r["architecture_family_id"] for r in train_ft} & {r["architecture_family_id"] for r in test + val + novel}
    assert not {r["struct_id"] for r in train_ft} & holdout
    assert not set(exemplars) & ({r["sample_id"] for r in test + novel + val})

    splits = {"seed": SEED, "networkx_version": nx.__version__,
              "holdout_struct_ids": sorted(holdout), "holdout_log": holdout_log,
              "few_shot_exemplars": exemplars,
              "sizes": {"train_ft": len(train_ft), "val": len(val), "novel_topo": len(novel), "test": len(test),
                        "test_anon": len(anon), "test_mini": len(mini),
                        "test_struct_unseen": sum(1 for r in test if not r["struct_seen"])},
              "train_ft": sorted(ft_ids), "val": sorted(r["sample_id"] for r in val),
              "novel_topo": [r["sample_id"] for r in novel], "test_mini": [r["sample_id"] for r in mini],
              "struct_id": {r["sample_id"]: r["struct_id"] for r in recs},
              "struct_id_kind": {r["sample_id"]: r["struct_id_kind"] for r in recs}}
    with open(os.path.join(PROC_DIR, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)
    print(json.dumps(splits["sizes"]), "holdout k per tier:", {t: holdout_log[t]["k"] for t in TIERS})

    # 7 + 9. SFT files and token audit
    if args.tokenizer and os.path.isdir(args.tokenizer):
        build_sft(args, train_ft, val, by_id, exemplars)
    else:
        print(f"[skip] SFT files: tokenizer dir not found: {args.tokenizer}")

    # 10. stats
    write_stats(recs, splits, holdout_log)


def build_sft(args, train_ft, val, by_id, exemplars):
    from transformers import AutoTokenizer

    from tgraph.prompts import build_messages, render, turn_end_suffix
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    suffix = turn_end_suffix(tok)
    end_tok = suffix.rstrip("\n")
    ids = tok(end_tok, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1 and end_tok in tok.all_special_tokens + list(tok.get_added_vocab()), (suffix, ids)
    print(f"turn-end token: {end_tok!r} (id {ids[0]}); full suffix {suffix!r}")
    for other in [d for d in args.extra_tokenizers.split(",") if d]:
        t2 = AutoTokenizer.from_pretrained(other)
        m = build_messages("class A(nn.Module):\n    pass")
        same = render(t2, m) == render(tok, m) and turn_end_suffix(t2) == suffix
        print(f"chat template identical to {os.path.basename(other)}: {same}")
        assert same, other
    stats = {}
    for name, rows in [("train", train_ft), ("val", val)]:
        out, lens = [], []
        for r in rows:
            prompt = render(tok, build_messages(r["source_code"]))
            completion = dumps_cg(canonical_to_cg(r["target_graph"])) + end_tok
            out.append({"prompt": prompt, "completion": completion, "sample_id": r["sample_id"], "tier": r["tier"]})
            lens.append(len(tok(prompt + completion, add_special_tokens=False)["input_ids"]))
        write_jsonl(os.path.join(PROC_DIR, f"sft_{name}.jsonl"), out)
        lens.sort()
        stats[name] = {"n": len(lens), "max": lens[-1], "p99": lens[int(0.99 * (len(lens) - 1))],
                       "mean": round(sum(lens) / len(lens), 1)}
    ex = [(by_id[i]["source_code"], by_id[i]["target_graph"]) for i in exemplars]
    fs = [len(tok(render(tok, build_messages(r["source_code"], ex)), add_special_tokens=False)["input_ids"])
          for r in list(by_id.values())]
    stats["few_shot_prompt"] = {"max": max(fs), "mean": round(sum(fs) / len(fs), 1)}
    mx = max(stats["train"]["max"], stats["val"]["max"])
    stats["max_length"] = ((mx + 127) // 128) * 128 + 64
    assert stats["max_length"] >= mx                 # zero truncation by construction
    stats["canonical_prompt_example"] = render(tok, build_messages(by_id["L0_0000"]["source_code"]))
    stats["turn_end"] = end_tok
    with open(os.path.join(PROC_DIR, "token_audit.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("token audit:", {k: v for k, v in stats.items() if k != "canonical_prompt_example"})


def write_stats(recs, splits, holdout_log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = os.path.join(REPO, "results", "tables")
    fig_dir = os.path.join(REPO, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    L = ["# Dataset statistics\n", "Generated by `scripts/prepare_data.py` (seed 643).\n",
         "## Per tier\n", "| Tier | samples | train | test | nodes min/med/max | edges min/med/max | "
         "distinct structures | distinct kind-labelled structures |", "|---|---|---|---|---|---|---|---|"]
    for t in TIERS:
        rs = [r for r in recs if r["tier"] == t]
        nn_ = sorted(r["n_nodes"] for r in rs)
        ne = sorted(r["n_edges"] for r in rs)
        L.append(f"| {t} | {len(rs)} | {sum(r['split'] == 'train' for r in rs)} | {sum(r['split'] == 'test' for r in rs)} | "
                 f"{nn_[0]}/{nn_[len(nn_) // 2]}/{nn_[-1]} | {ne[0]}/{ne[len(ne) // 2]}/{ne[-1]} | "
                 f"{len({r['struct_id'] for r in rs})} | {len({r['struct_id_kind'] for r in rs})} |")
    L.append(f"\nTotal distinct unlabelled structures: **{len({r['struct_id'] for r in recs})}**; "
             f"kind-labelled: {len({r['struct_id_kind'] for r in recs})}.")
    test_structs = {r["struct_id"] for r in recs if r["split"] == "test"}
    train_structs = {r["struct_id"] for r in recs if r["split"] == "train"}
    L.append(f"Official-test structures also present in train: {len(test_structs & train_structs)}/{len(test_structs)} "
             "(the official test measures unseen code with seen topology).\n")
    L.append("## Splits\n")
    L.append("| split | samples |\n|---|---|")
    L += [f"| {k} | {v} |" for k, v in splits["sizes"].items()]
    L.append("\n## Novel-topology holdout\n")
    L.append("| Tier | structure classes in train | held out | held-out class sizes (samples) |\n|---|---|---|---|")
    for t in TIERS:
        h = holdout_log[t]
        L.append(f"| {t} | {h['n_classes']} | {h['k']} | {list(h['chosen'].values())} |")
    kinds = collections.Counter(n["kind"] for r in recs for n in r["target_graph"]["nodes"])
    L.append("\n## Kind vocabulary (node counts over all 5,000 graphs)\n")
    L.append(", ".join(f"`{k}` {v}" for k, v in kinds.most_common()))
    L.append(f"\n\nFew-shot exemplars (train_ft, augmentation none): {splits['few_shot_exemplars']}")
    L.append("\n![node and edge histograms](../figures/dataset_sizes.png)")
    with open(os.path.join(out_dir, "dataset_stats.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(PROC_DIR, "dataset_stats.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, lab in ((axes[0], "n_nodes", "nodes per graph"), (axes[1], "n_edges", "edges per graph")):
        for t in TIERS:
            ax.hist([r[key] for r in recs if r["tier"] == t], bins=range(4, 32), alpha=0.6, label=t)
        ax.set_xlabel(lab)
        ax.set_ylabel("samples")
    axes[0].legend(title="tier")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dataset_sizes.png"), dpi=150)
    print("wrote results/tables/dataset_stats.md")


if __name__ == "__main__":
    main()
