import copy
import random

from tgraph.metrics import compute_metrics, true_ged
from tgraph.properties import ALL_PROPS


def test_identity_perfect(records):
    for r in random.Random(0).sample(records, 50):
        m = compute_metrics(r["target_graph"], r["target_graph"])
        for k in ["node_f1", "edge_f1", "exact_strict", "isomorphic", "graph_similarity", "pps", "shape_acc",
                  "reach_f1", "edge_f1_aligned", "kind_multiset_f1", "typed_edge_multiset_f1"]:
            assert m[k] == 1.0, (r["sample_id"], k, m[k])
        assert m["aed"] == 0


def _l2(records):
    return next(r for r in records if r["tier"] == "L2")["target_graph"]


def test_corruptions_lower_scores(records):
    g = _l2(records)
    c = copy.deepcopy(g)
    c["edges"].pop(3)
    m = compute_metrics(c, g)
    assert m["edge_f1"] < 1 and m["exact_strict"] == 0 and m["isomorphic"] == 0
    c = copy.deepcopy(g)
    c["nodes"][1]["kind"] = "Linear"
    m = compute_metrics(c, g)
    assert m["node_f1"] < 1 and m["exact_strict"] == 0


def test_permutation_and_renaming_keep_isomorphism(records):
    g = _l2(records)
    perm = list(range(len(g["nodes"])))
    random.Random(1).shuffle(perm)
    new_id = {n["id"]: f"node_{perm[i]}" for i, n in enumerate(g["nodes"])}
    nodes = [dict(n, id=new_id[n["id"]], name=f"renamed_{i}" if n["op"] != "placeholder" else n["name"])
             for i, n in enumerate(g["nodes"])]
    nodes.sort(key=lambda n: int(n["id"].split("_")[1]))
    p = {"nodes": nodes, "edges": [{"src": new_id[e["src"]], "dst": new_id[e["dst"]]} for e in g["edges"]]}
    m = compute_metrics(p, g)
    assert m["isomorphic"] == 1.0 and m["exact_strict"] == 0.0 and m["edge_f1"] < 1.0
    # every property survives except hierarchy_groups, which is defined by name prefixes
    assert all(m[f"prop_{k}"] == 1.0 for k in ALL_PROPS if k != "hierarchy_groups")


def test_parse_failure_is_zero(records):
    m = compute_metrics(None, _l2(records))
    assert m["edge_f1"] == 0 and m["graph_similarity"] == 0


def test_aed_upper_bounds_ged(records):
    rng = random.Random(3)
    small = [r["target_graph"] for r in records if len(r["target_graph"]["nodes"]) <= 9]
    checked = 0
    for _ in range(40):
        a, b = rng.sample(small, 2)
        ged, exact = true_ged(a, b, timeout=2.0)
        if ged is None or not exact:
            continue
        assert compute_metrics(a, b)["aed"] >= ged - 1e-9
        checked += 1
        if checked >= 20:
            break
    assert checked >= 5
