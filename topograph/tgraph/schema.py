"""Compact graph JSON (CG-JSON, what the LLM emits) <-> canonical dataset format, plus the JSON schema.

CG:        {"nodes": [{name, op, kind, shape}], "edges": [[i, j], ...]}      (ids = list positions)
canonical: {"nodes": [{id, name, op, kind, scope, shape}], "edges": [{src, dst}]}
"""
import json
import os

import networkx as nx

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "graph_schema.json")
with open(SCHEMA_PATH) as _f:
    SCHEMA = json.load(_f)

NODE_KEYS = ("name", "op", "kind", "shape")


def canonical_to_cg(g):
    idx = {n["id"]: i for i, n in enumerate(g["nodes"])}
    return {"nodes": [{k: n[k] for k in NODE_KEYS} for n in g["nodes"]],
            "edges": [[idx[e["src"]], idx[e["dst"]]] for e in g["edges"]]}


def cg_to_canonical(cg, scope_map=None):
    """scope_map: optional {node_name: dotted module path} (from codeinfo) for nested submodules."""
    scope_map = scope_map or {}
    nodes = []
    for i, n in enumerate(cg["nodes"]):
        scope = scope_map.get(n["name"], n["name"]) if n["op"] == "call_module" else ""
        nodes.append({"id": f"node_{i}", "name": n["name"], "op": n["op"], "kind": n["kind"],
                      "scope": scope, "shape": n.get("shape")})
    edges = [{"src": f"node_{a}", "dst": f"node_{b}"} for a, b in cg["edges"]]
    return {"nodes": nodes, "edges": edges}


def dumps_cg(cg):
    """The one serialization used for training targets and few-shot exemplars."""
    return json.dumps(cg, separators=(",", ":"))


def to_nx(g):
    """Canonical graph -> DiGraph keyed by node id, with node attributes (index, name, op, kind, shape)."""
    G = nx.DiGraph()
    for i, n in enumerate(g["nodes"]):
        G.add_node(n["id"], index=i, name=n["name"], op=n["op"], kind=n["kind"], shape=n.get("shape"))
    for e in g["edges"]:
        if e["src"] in G and e["dst"] in G:
            G.add_edge(e["src"], e["dst"])
    return G


def normalize_placeholder(g):
    """If a graph has a single placeholder, its name becomes 'x' (dataset convention). Returns a copy."""
    g = {"nodes": [dict(n) for n in g["nodes"]], "edges": [dict(e) for e in g["edges"]]}
    ph = [n for n in g["nodes"] if n["op"] == "placeholder"]
    if len(ph) == 1:
        ph[0]["name"] = "x"
    return g
