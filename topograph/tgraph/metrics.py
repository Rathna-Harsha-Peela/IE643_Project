"""Per-sample metrics between a predicted and a reference graph (both canonical format)."""
from collections import Counter

import networkx as nx

from .align import align
from .properties import ALL_PROPS, COUNT_PROPS, compute_properties
from .schema import normalize_placeholder, to_nx

HEADLINE = ["fully_valid", "edge_f1", "exact_strict", "isomorphic", "graph_similarity", "pps", "shape_acc"]


def _prf(tp, n_pred, n_ref):
    p = tp / n_pred if n_pred else (1.0 if n_ref == 0 else 0.0)
    r = tp / n_ref if n_ref else (1.0 if n_pred == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _multiset_f1(a, b):
    ca, cb = Counter(a), Counter(b)
    tp = sum((ca & cb).values())
    return _prf(tp, sum(ca.values()), sum(cb.values()))[2]


def _reach_pairs(G):
    return {(u, v) for u in G for v in nx.descendants(G, u)}


def empty_metrics(ref):
    """Metrics for an unparseable prediction: everything zero, properties unmatched."""
    m = {k: 0.0 for k in ["node_p", "node_r", "node_f1", "node_f1_aligned", "kind_multiset_f1", "edge_p", "edge_r",
                          "edge_f1", "edge_f1_aligned", "typed_edge_multiset_f1", "exact_strict", "isomorphic",
                          "wl_equal", "shape_acc", "graph_similarity", "pps", "reach_p", "reach_r", "reach_f1"]}
    rp = compute_properties(ref)
    m["aed"] = len(ref["nodes"]) + len(ref["edges"])
    m["n_pred_nodes"], m["n_ref_nodes"] = 0, len(ref["nodes"])
    for k in ALL_PROPS:
        m[f"prop_{k}"] = 0.0
    m["ref_props"], m["pred_props"] = rp, None
    return m


def compute_metrics(pred, ref):
    """pred may be None (parse failure)."""
    ref = normalize_placeholder(ref)
    if pred is None or not pred.get("nodes"):
        return empty_metrics(ref)
    pred = normalize_placeholder(pred)
    pn = {n["id"]: n for n in pred["nodes"]}
    rn = {n["id"]: n for n in ref["nodes"]}
    strict, aligned = align(pred, ref)
    m = {"n_pred_nodes": len(pn), "n_ref_nodes": len(rn)}

    # nodes
    tp = sum(1 for p, r in strict.items() if (pn[p]["op"], pn[p]["kind"]) == (rn[r]["op"], rn[r]["kind"]))
    m["node_p"], m["node_r"], m["node_f1"] = _prf(tp, len(pn), len(rn))
    tpa = sum(1 for p, r in aligned.items() if pn[p]["kind"] == rn[r]["kind"])
    m["node_f1_aligned"] = _prf(tpa, len(pn), len(rn))[2]
    m["kind_multiset_f1"] = _multiset_f1([n["kind"] for n in pred["nodes"]], [n["kind"] for n in ref["nodes"]])

    # edges
    pe = {(pn[e["src"]]["name"], pn[e["dst"]]["name"]) for e in pred["edges"] if e["src"] in pn and e["dst"] in pn}
    re_ = {(rn[e["src"]]["name"], rn[e["dst"]]["name"]) for e in ref["edges"]}
    m["edge_p"], m["edge_r"], m["edge_f1"] = _prf(len(pe & re_), len(pe), len(re_))
    pe_ids = {(e["src"], e["dst"]) for e in pred["edges"] if e["src"] in pn and e["dst"] in pn}
    re_ids = {(e["src"], e["dst"]) for e in ref["edges"]}
    mapped = {(aligned[a], aligned[b]) for a, b in pe_ids if a in aligned and b in aligned}
    m["edge_f1_aligned"] = _prf(len(mapped & re_ids), len(pe_ids), len(re_ids))[2]
    m["typed_edge_multiset_f1"] = _multiset_f1([(pn[a]["kind"], pn[b]["kind"]) for a, b in pe_ids],
                                               [(rn[a]["kind"], rn[b]["kind"]) for a, b in re_ids])

    # exact / isomorphic
    node_sig = lambda g: Counter((n["name"], n["op"], n["kind"]) for n in g["nodes"])  # noqa: E731
    m["exact_strict"] = float(node_sig(pred) == node_sig(ref) and pe == re_)
    Gp, Gr = to_nx(pred), to_nx(ref)
    same_size = Gp.number_of_nodes() == Gr.number_of_nodes() and Gp.number_of_edges() == Gr.number_of_edges()
    m["isomorphic"] = float(same_size and nx.is_isomorphic(Gp, Gr, node_match=lambda a, b: a["kind"] == b["kind"]))
    m["wl_equal"] = float(nx.weisfeiler_lehman_graph_hash(Gp, node_attr="kind", iterations=3) ==
                          nx.weisfeiler_lehman_graph_hash(Gr, node_attr="kind", iterations=3))

    # shapes
    m["shape_acc"] = (sum(1 for p, r in strict.items() if pn[p].get("shape") == rn[r].get("shape")) / len(strict)
                      if strict else 0.0)

    # aligned edit distance
    node_cost = (len(pn) - len(aligned)) + (len(rn) - len(aligned)) + \
        sum(1 for p, r in aligned.items() if pn[p]["kind"] != rn[r]["kind"])
    unmapped_pred_edges = len(pe_ids) - len(mapped)       # edges touching unmatched nodes count as differences
    edge_cost = len(re_ids ^ mapped) + unmapped_pred_edges
    m["aed"] = node_cost + edge_cost
    denom = len(pn) + len(rn) + len(pe_ids) + len(re_ids)
    m["graph_similarity"] = 1 - min(1.0, m["aed"] / denom) if denom else 1.0

    # properties
    pp, rp = compute_properties(pred), compute_properties(ref)
    for k in ALL_PROPS:
        m[f"prop_{k}"] = float(pp[k] == rp[k])
    m["pps"] = sum(m[f"prop_{k}"] for k in ALL_PROPS) / len(ALL_PROPS)
    for k in COUNT_PROPS:
        m[f"abs_err_{k}"] = abs(pp[k] - rp[k])
    m["pred_props"], m["ref_props"] = pp, rp

    # transitive connectivity under the alignment
    rr = _reach_pairs(Gr)
    pr = {(aligned[a], aligned[b]) for a, b in _reach_pairs(Gp) if a in aligned and b in aligned}
    n_pred_reach = len(_reach_pairs(Gp))
    m["reach_p"], m["reach_r"], m["reach_f1"] = _prf(len(pr & rr), n_pred_reach, len(rr))
    return m


# ---- true graph edit distance (slow; only on a small subset) ---------------------------------------------------
def true_ged(pred, ref, timeout=2.0):
    """Best GED found within `timeout` seconds -> (value, exact_flag). exact_flag False = upper bound (timed out)."""
    import time
    Gp, Gr = to_nx(normalize_placeholder(pred)), to_nx(normalize_placeholder(ref))
    t0 = time.time()
    v = nx.graph_edit_distance(Gp, Gr, node_subst_cost=lambda a, b: 0 if a["kind"] == b["kind"] else 1,
                               node_del_cost=lambda a: 1, node_ins_cost=lambda a: 1, edge_subst_cost=lambda a, b: 0,
                               edge_del_cost=lambda a: 1, edge_ins_cost=lambda a: 1, timeout=timeout)
    return v, (time.time() - t0) < 0.95 * timeout
