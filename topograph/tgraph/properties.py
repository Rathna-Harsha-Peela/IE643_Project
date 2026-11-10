"""Topological properties of a canonical graph (used for the Property Preservation Score)."""
import itertools
import re

import networkx as nx

from .schema import to_nx

PATH_CAP = 10_000
CYCLE_CAP = 100
COUNT_PROPS = ["n_nodes", "n_edges", "n_sources", "n_sinks", "n_branch", "n_merge", "depth", "width",
               "n_skip_edges", "n_paths_in_out", "n_cycles", "hierarchy_groups"]
BOOL_PROPS = ["is_dag", "weakly_connected"]
ALL_PROPS = COUNT_PROPS + BOOL_PROPS


def name_prefix(node):
    """Cluster key: dotted scope head, else the part of the name before the first '_' (only for call_module)."""
    if node.get("op") != "call_module":
        return None
    scope = node.get("scope") or ""
    if "." in scope:
        return scope.split(".")[0]
    m = re.match(r"^([A-Za-z0-9]+)_", node["name"])
    return m.group(1) if m else None


def hierarchy_groups(g):
    """{prefix: [node ids]} for prefixes shared by >= 2 call_module nodes."""
    groups = {}
    for n in g["nodes"]:
        p = name_prefix(n)
        if p:
            groups.setdefault(p, []).append(n["id"])
    return {k: v for k, v in groups.items() if len(v) >= 2}


def levels(G):
    """Longest-path level from the sources for every node of a DAG."""
    lv = {}
    for u in nx.topological_sort(G):
        lv[u] = max((lv[p] + 1 for p in G.predecessors(u)), default=0)
    return lv


def compute_properties(g):
    G = to_nx(g)
    dag = nx.is_directed_acyclic_graph(G)
    p = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
         "n_sources": sum(1 for u in G if G.in_degree(u) == 0), "n_sinks": sum(1 for u in G if G.out_degree(u) == 0),
         "n_branch": sum(1 for u in G if G.out_degree(u) > 1), "n_merge": sum(1 for u in G if G.in_degree(u) > 1),
         "is_dag": dag, "weakly_connected": G.number_of_nodes() > 0 and nx.is_weakly_connected(G),
         "hierarchy_groups": len(hierarchy_groups(g))}
    if dag and G.number_of_nodes():
        lv = levels(G)
        p["depth"] = max(lv.values())
        p["width"] = max(len(gen) for gen in nx.topological_generations(G))
        p["n_skip_edges"] = sum(1 for u, v in G.edges if lv[v] - lv[u] > 1)
        npaths = {}
        srcs = [n["id"] for n in g["nodes"] if n["op"] == "placeholder"]
        for u in nx.topological_sort(G):
            npaths[u] = min(PATH_CAP, (1 if u in srcs else 0) + sum(npaths[q] for q in G.predecessors(u)))
        outs = [n["id"] for n in g["nodes"] if n["op"] == "output"]
        p["n_paths_in_out"] = min(PATH_CAP, sum(npaths[o] for o in outs))
        p["n_cycles"] = 0
    else:
        p.update(depth=-1, width=-1, n_skip_edges=-1, n_paths_in_out=-1)
        p["n_cycles"] = sum(1 for _ in itertools.islice(nx.simple_cycles(G), CYCLE_CAP)) if G.number_of_nodes() else 0
    return p
