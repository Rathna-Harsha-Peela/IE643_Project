"""Graph rendering: graphviz (preferred) with a matplotlib fallback when the `dot` binary is missing.

render(graph, path, ...)          one graph, hierarchy clusters, optional neuron expansion
render_diff(pred, ref, path)      union graph under the alignment, colour-coded differences + legend
render_pair(pred, ref, path)      reference and prediction side by side
Graphs are canonical (dataset format). `path` is given without extension; the written file path is returned.
"""
import os
import shutil

import networkx as nx

from .align import align
from .properties import hierarchy_groups
from .schema import normalize_placeholder, to_nx

GREY, BLUE, WHITE, RED, ORANGE = "#d9d9d9", "#cfe2f3", "#ffffff", "#cc0000", "#e69138"
FONT = "10"


def has_dot():
    return shutil.which("dot") is not None


def _shape_str(s):
    return "[" + ",".join(str(v) for v in s) + "]" if isinstance(s, list) and s else ""


def _style(n):
    if n["op"] in ("placeholder", "output"):
        return {"shape": "ellipse", "style": "filled", "fillcolor": GREY}
    if n["op"] == "call_module":
        return {"shape": "box", "style": "filled", "fillcolor": BLUE}
    return {"shape": "box", "style": "rounded,filled", "fillcolor": WHITE}


def _label(n, show_shapes=True, shape_color=None, kind_color=None):
    """HTML-like label so individual parts (kind, shape) can be coloured in the diff view."""
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    parts = [f"<B>{esc(n['name'])}</B>"]
    if n["op"] == "call_module":
        k = esc(n["kind"])
        parts.append(f'<FONT COLOR="{kind_color}">{k}</FONT>' if kind_color else k)
    elif n["op"] not in ("placeholder", "output") and n["kind"] != n["name"].rstrip("_0123456789"):
        parts.append(f"<I>{esc(n['kind'])}</I>")
    if show_shapes and _shape_str(n.get("shape")):
        s = esc(_shape_str(n.get("shape")))
        parts.append(f'<FONT COLOR="{shape_color}">{s}</FONT>' if shape_color else s)
    return "<" + "<BR/>".join(parts) + ">"


def _clusters(g):
    """{prefix: set(node ids)}: call_module groups plus functional nodes whose unique predecessor is clustered."""
    groups = {k: set(v) for k, v in hierarchy_groups(g).items()}
    where = {nid: k for k, ids in groups.items() for nid in ids}
    preds = {}
    for e in g["edges"]:
        preds.setdefault(e["dst"], []).append(e["src"])
    for n in g["nodes"]:                                   # nodes are in execution order, so chains propagate
        if n["op"] in ("call_function", "call_method") and n["id"] not in where:
            ps = preds.get(n["id"], [])
            if len(ps) == 1 and ps[0] in where:
                where[n["id"]] = where[ps[0]]
                groups[where[ps[0]]].add(n["id"])
    return groups


def _expand_neurons(g, max_units=8):
    """Display-only: Linear layers with in,out <= max_units become unit nodes with complete bipartite edges."""
    nodes, edges = [], []
    units = {}                                             # node id -> list of unit ids
    for n in g["nodes"]:
        s = n.get("shape") or []
        out = s[-1] if s else None
        if n["kind"] == "Linear" and isinstance(out, int) and out <= max_units:
            ids = [f"{n['id']}_u{k}" for k in range(out)]
            units[n["id"]] = ids
            for k, uid in enumerate(ids):
                nodes.append({**n, "id": uid, "name": f"{n['name']}_u{k}" if out > 1 else n["name"], "shape": [1, 1]})
        elif n["op"] == "placeholder" and isinstance(out, int) and out <= max_units and len(s) == 2:
            ids = [f"{n['id']}_u{k}" for k in range(out)]
            units[n["id"]] = ids
            for k, uid in enumerate(ids):
                nodes.append({**n, "id": uid, "name": f"{n['name']}{k}", "shape": [1, 1]})
        else:
            nodes.append(n)
    for e in g["edges"]:
        srcs = units.get(e["src"], [e["src"]])
        dsts = units.get(e["dst"], [e["dst"]])
        if e["dst"] in units:                              # into a Linear: complete bipartite
            edges += [{"src": a, "dst": b} for a in srcs for b in dsts]
        else:
            edges += [{"src": a, "dst": dsts[0]} for a in srcs]
    return {"nodes": nodes, "edges": edges}


def _dot_graph(title=None):
    import graphviz
    d = graphviz.Digraph(graph_attr={"rankdir": "TB", "fontsize": "12", "labelloc": "t", "nodesep": "0.25",
                                     "ranksep": "0.35"},
                         node_attr={"fontsize": FONT, "fontname": "Helvetica", "margin": "0.08,0.03"},
                         edge_attr={"arrowsize": "0.6"})
    if title:
        d.attr(label=title)
    return d


def _add_graph(d, g, prefix="", cluster=True, show_shapes=True, highlight=None, node_extra=None):
    highlight = set(highlight or [])
    node_extra = node_extra or {}
    groups = _clusters(g) if cluster else {}
    in_cluster = {nid: k for k, ids in groups.items() for nid in ids}
    for k, ids in groups.items():
        with d.subgraph(name=f"cluster_{prefix}{k}") as c:
            c.attr(label=k, style="dashed", color="#888888", fontsize=FONT)
            for n in g["nodes"]:
                if n["id"] in ids:
                    c.node(prefix + n["id"], **_node_attrs(n, show_shapes, highlight, node_extra))
    for n in g["nodes"]:
        if n["id"] not in in_cluster:
            d.node(prefix + n["id"], **_node_attrs(n, show_shapes, highlight, node_extra))
    for e in g["edges"]:
        d.edge(prefix + e["src"], prefix + e["dst"])


def _node_attrs(n, show_shapes, highlight, node_extra):
    a = _style(n)
    extra = node_extra.get(n["id"], {})
    a["label"] = _label(n, show_shapes, extra.pop("_shape_color", None), extra.pop("_kind_color", None))
    if n["id"] in highlight:
        a.update(color=RED, penwidth="2")
    a.update(extra)
    return a


def _write(d, path, fmt):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return d.render(filename=path, format=fmt, cleanup=True)


def render(graph, path, title=None, fmt="svg", highlight=None, cluster=True, expand_neurons=False,
           show_shapes=True, backend=None):
    g = _expand_neurons(graph) if expand_neurons else graph
    if (backend or ("dot" if has_dot() else "mpl")) == "mpl":
        return render_mpl(g, path, title=title, fmt=fmt if fmt != "svg" else "svg")
    d = _dot_graph(title)
    _add_graph(d, g, cluster=cluster and not expand_neurons, show_shapes=show_shapes, highlight=highlight)
    return _write(d, path, fmt)


def diff_graph(pred, ref):
    """Union graph under the alignment. Returns (graph, node_status, edge_status)."""
    pred, ref = normalize_placeholder(pred), normalize_placeholder(ref)
    _, pi = align(pred, ref)
    pn = {n["id"]: n for n in pred["nodes"]}
    rn = {n["id"]: n for n in ref["nodes"]}
    nodes, status = [], {}
    inv = {r: p for p, r in pi.items()}
    for n in ref["nodes"]:
        p = inv.get(n["id"])
        nid = "r_" + n["id"]
        if p is None:
            status[nid] = {"state": "missing"}
            nodes.append({**n, "id": nid})
        else:
            pnode = pn[p]
            status[nid] = {"state": "matched", "wrong_kind": pnode["kind"] != n["kind"],
                           "wrong_shape": pnode.get("shape") != n.get("shape")}
            nodes.append({**pnode, "id": nid, "_ref": n})
    for n in pred["nodes"]:
        if n["id"] not in pi:
            nid = "p_" + n["id"]
            status[nid] = {"state": "hallucinated"}
            nodes.append({**n, "id": nid})
    to_union_p = {p: "r_" + r for p, r in pi.items()}
    to_union_p.update({n["id"]: "p_" + n["id"] for n in pred["nodes"] if n["id"] not in pi})
    re_ = {("r_" + e["src"], "r_" + e["dst"]) for e in ref["edges"]}
    pe = {(to_union_p[e["src"]], to_union_p[e["dst"]]) for e in pred["edges"] if e["src"] in pn and e["dst"] in pn}
    edges, estatus = [], {}
    for e in sorted(re_ | pe):
        estatus[e] = "matched" if (e in re_ and e in pe) else ("missing" if e in re_ else "hallucinated")
        edges.append({"src": e[0], "dst": e[1]})
    return {"nodes": nodes, "edges": edges}, status, estatus


def render_diff(pred, ref, path, title=None, fmt="svg", show_shapes=True, legend=True):
    g, st, est = diff_graph(pred, ref)
    if not has_dot():
        return render_mpl(g, path, title=title, fmt=fmt, node_status=st, edge_status=est)
    d = _dot_graph(title or "Diff: prediction vs reference")
    extra = {}
    for n in g["nodes"]:
        s = st[n["id"]]
        if s["state"] == "missing":
            extra[n["id"]] = {"color": RED, "style": "dashed", "fontcolor": RED, "penwidth": "1.5"}
        elif s["state"] == "hallucinated":
            extra[n["id"]] = {"fillcolor": ORANGE, "style": "filled" + (",rounded" if n["op"] != "call_module" else "")}
        else:
            if s["wrong_kind"]:
                extra[n["id"]] = {"_kind_color": RED, "color": RED}
            if s["wrong_shape"]:
                extra.setdefault(n["id"], {})["_shape_color"] = RED
    for n in g["nodes"]:
        d.node(n["id"], **_node_attrs(n, show_shapes, set(), extra))
    for e in g["edges"]:
        s = est[(e["src"], e["dst"])]
        attrs = {"matched": {"color": "black"}, "missing": {"color": RED, "style": "dashed"},
                 "hallucinated": {"color": ORANGE, "penwidth": "2.2"}}[s]
        d.edge(e["src"], e["dst"], **attrs)
    legend_html = ('<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">'
              '<TR><TD COLSPAN="2"><B>Legend</B></TD></TR>'
              '<TR><TD>black edge</TD><TD>matched</TD></TR>'
              f'<TR><TD><FONT COLOR="{RED}">red dashed</FONT></TD><TD>missing (reference only)</TD></TR>'
              f'<TR><TD><FONT COLOR="{ORANGE}">orange bold / fill</FONT></TD><TD>hallucinated (prediction only)</TD></TR>'
              f'<TR><TD><FONT COLOR="{RED}">red text</FONT></TD><TD>wrong kind / shape</TD></TR></TABLE>>')
    if legend:
        d.node("legend", label=legend_html, shape="plaintext", fontsize="9")
    return _write(d, path, fmt)


def render_pair(pred, ref, path, model_name="model", fmt="svg", show_shapes=True):
    if not has_dot():
        return render_mpl_pair(pred, ref, path, model_name, fmt)
    d = _dot_graph()
    for tag, g, title in (("r_", ref, "Reference (torch.fx)"), ("p_", pred, f"Predicted ({model_name})")):
        with d.subgraph(name=f"cluster_side_{tag}") as c:
            c.attr(label=title, style="solid", color="#444444", fontsize="12")
            _add_graph(c, g, prefix=tag, cluster=True, show_shapes=show_shapes)
    return _write(d, path, fmt)


# ---- matplotlib fallback ---------------------------------------------------------------------------------------
def _mpl_layout(g):
    G = to_nx(g)
    if nx.is_directed_acyclic_graph(G):
        for i, gen in enumerate(nx.topological_generations(G)):
            for u in gen:
                G.nodes[u]["layer"] = i
    else:
        for u in G:
            G.nodes[u]["layer"] = G.nodes[u]["index"]
    pos = nx.multipartite_layout(G, subset_key="layer")      # layers along x ...
    return G, {u: (p[1], -p[0]) for u, p in pos.items()}       # ... rotated so layers run top to bottom


def render_mpl(g, path, title=None, fmt="png", node_status=None, edge_status=None, ax=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G, pos = _mpl_layout(g)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, max(4, 0.55 * (max(nx.get_node_attributes(G, "layer").values(), default=0) + 2))))
    byid = {n["id"]: n for n in g["nodes"]}
    colors, edgecols = [], []
    for u in G:
        n = byid[u]
        c = GREY if n["op"] in ("placeholder", "output") else (BLUE if n["op"] == "call_module" else WHITE)
        st = (node_status or {}).get(u, {})
        if st.get("state") == "hallucinated":
            c = ORANGE
        colors.append(c)
        edgecols.append(RED if st.get("state") == "missing" or st.get("wrong_kind") else "#333333")
    ecol = [{"matched": "black", "missing": RED, "hallucinated": ORANGE}.get((edge_status or {}).get(e), "black")
            for e in G.edges]
    estyle = ["dashed" if (edge_status or {}).get(e) == "missing" else "solid" for e in G.edges]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=ecol, style=estyle, arrows=True, arrowsize=10,
                           node_size=1400, width=[2.2 if c == ORANGE else 1.0 for c in ecol])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, edgecolors=edgecols, node_shape="s", node_size=1400)
    labels = {u: byid[u]["name"] + ("\n" + byid[u]["kind"] if byid[u]["op"] == "call_module" else "") +
              ("\n" + _shape_str(byid[u].get("shape")) if _shape_str(byid[u].get("shape")) else "") for u in G}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6)
    ax.set_title(title or "", fontsize=10)
    ax.axis("off")
    if own:
        out = f"{path}.{fmt}"
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out
    return None


def render_mpl_pair(pred, ref, path, model_name="model", fmt="png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 8))
    render_mpl(ref, None, title="Reference (torch.fx)", ax=axes[0])
    render_mpl(pred, None, title=f"Predicted ({model_name})", ax=axes[1])
    out = f"{path}.{fmt}"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
