"""Node alignment between predicted and reference canonical graphs: exact name match, then Hungarian on the rest."""
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import to_nx


def _levels(G):
    if not nx.is_directed_acyclic_graph(G):
        return {u: G.nodes[u]["index"] for u in G}            # fallback: list position
    lv = {}
    for u in nx.topological_sort(G):
        lv[u] = max((lv[p] + 1 for p in G.predecessors(u)), default=0)
    return lv


def align(pred, ref):
    """Return (strict, aligned): dicts pred_id -> ref_id. strict = name matches only."""
    Gp, Gr = to_nx(pred), to_nx(ref)
    ref_by_name = {}
    for n in ref["nodes"]:
        ref_by_name.setdefault(n["name"], n["id"])
    strict, used = {}, set()
    for n in pred["nodes"]:
        rid = ref_by_name.get(n["name"])
        if rid is not None and rid not in used:
            strict[n["id"]] = rid
            used.add(rid)
    aligned = dict(strict)
    pu = [n for n in pred["nodes"] if n["id"] not in strict]
    ru = [n for n in ref["nodes"] if n["id"] not in used]
    if pu and ru:
        lp, lr = _levels(Gp), _levels(Gr)
        big = 1e6
        C = np.full((len(pu), len(ru)), big)
        for i, a in enumerate(pu):
            for j, b in enumerate(ru):
                if a["kind"] == b["kind"]:
                    C[i, j] = 0.25 * abs(lp[a["id"]] - lr[b["id"]])
        rows, cols = linear_sum_assignment(C)
        for i, j in zip(rows, cols):
            if C[i, j] < 1:
                aligned[pu[i]["id"]] = ru[j]["id"]
    return strict, aligned
