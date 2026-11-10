"""Rule-based failure taxonomy for one prediction (a sample can get several labels)."""
from .align import align
from .properties import levels
from .schema import normalize_placeholder, to_nx

LABELS = ["parse_fail", "missing_skip_edge", "missing_edge", "extra_edge", "missing_node", "hallucinated_node", "dead_code_included",
          "wrong_kind", "naming_only", "shape_only", "order_error", "cycle"]


def classify(pred, ref, m, validation):
    """pred/ref canonical (pred may be None); m = compute_metrics output; validation = ValidationReport dict."""
    if pred is None or not pred.get("nodes"):
        return ["parse_fail"]
    if m["exact_strict"] == 1.0 and m["shape_acc"] == 1.0:
        return []
    codes = {e["code"] for e in validation.get("errors", [])}
    labels = []
    if m["exact_strict"] == 1.0:
        return ["shape_only"]
    pred, ref = normalize_placeholder(pred), normalize_placeholder(ref)
    _, pi = align(pred, ref)
    Gr = to_nx(ref)
    lv = levels(Gr)
    pe = {(pi[e["src"]], pi[e["dst"]]) for e in pred["edges"] if e["src"] in pi and e["dst"] in pi}
    re_ = {(e["src"], e["dst"]) for e in ref["edges"]}
    missing = [(u, v) for u, v in re_ if (u, v) not in pe]
    if any(lv[v] - lv[u] > 1 for u, v in missing):
        labels.append("missing_skip_edge")
    if any(lv[v] - lv[u] <= 1 for u, v in missing):
        labels.append("missing_edge")
    if len(pe - re_) + sum(1 for e in pred["edges"] if not (e["src"] in pi and e["dst"] in pi)) > 0:
        labels.append("extra_edge")
    if len(pi) < len(ref["nodes"]):
        labels.append("missing_node")
    if len(pi) < len(pred["nodes"]):
        labels.append("hallucinated_node")
    if "DEAD_CODE_INCLUDED" in codes:
        labels.append("dead_code_included")
    pn = {n["id"]: n for n in pred["nodes"]}
    rn = {n["id"]: n for n in ref["nodes"]}
    names = {n["name"]: n for n in ref["nodes"]}
    if any(n["name"] in names and names[n["name"]]["kind"] != n["kind"] for n in pred["nodes"]) or \
       any(pn[p]["kind"] != rn[r]["kind"] for p, r in pi.items()):
        labels.append("wrong_kind")
    if m["isomorphic"] == 1.0:
        labels.append("naming_only")
    if "ORDER" in codes:
        labels.append("order_error")
    if "CYCLE" in codes:
        labels.append("cycle")
    return labels
