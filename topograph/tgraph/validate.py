"""Graph validity and topological-consistency checker (never uses the reference graph).

validate(cg, source_code=None, input_shape=None, parse_info=None) -> ValidationReport
Levels: V0 parse/schema, V1 structure, V2 arity, V3 code consistency (AST), V4 shapes.
"""
import math
import re
from dataclasses import asdict, dataclass, field

import networkx as nx

from .codeinfo import analyze

LEVEL_CODES = {
    "V0": ["PARSE", "SCHEMA"],
    "V1": ["DUP_NAME", "BAD_EDGE", "SELF_LOOP", "DUP_EDGE", "NO_INPUT", "OUTPUT_COUNT", "INPUT_HAS_PRED",
           "OUTPUT_DEGREE", "CYCLE", "ORDER", "ORPHAN", "DEAD_END", "OP_KIND"],
    "V2": ["ARITY"],
    "V3": ["UNDECLARED_MODULE", "KIND_MISMATCH", "MISSING_MODULE", "DEAD_CODE_INCLUDED"],
    "V4": ["SHAPE"],
}
CODE_LEVEL = {c: lv for lv, cs in LEVEL_CODES.items() for c in cs}

# layers that take exactly one tensor and (for SHAPE_PRESERVING) return the same shape
SINGLE_INPUT_MODULES = {"Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d", "BatchNorm1d", "BatchNorm2d", "LayerNorm",
                        "GroupNorm", "InstanceNorm2d", "ReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "LeakyReLU",
                        "ELU", "Softmax", "Dropout", "Dropout2d", "Identity", "Linear", "MaxPool2d", "AvgPool2d",
                        "AdaptiveAvgPool2d", "AdaptiveMaxPool2d", "Flatten", "Upsample", "Hardswish", "Mish"}
SHAPE_PRESERVING_MODULES = {"BatchNorm1d", "BatchNorm2d", "LayerNorm", "GroupNorm", "InstanceNorm2d", "ReLU", "GELU",
                            "SiLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU", "Softmax", "Dropout", "Dropout2d",
                            "Identity", "Hardswish", "Mish"}
SHAPE_PRESERVING_FUNCS = {"relu", "sigmoid", "tanh", "gelu", "silu", "dropout", "leaky_relu", "elu", "softmax",
                          "hardswish", "relu_", "contiguous"}
SINGLE_INPUT_FUNCS = SHAPE_PRESERVING_FUNCS | {"flatten"}
BINARY_FUNCS = {"add", "mul", "sub", "truediv", "iadd", "imul"}
MULTI_FUNCS = {"cat", "stack"}


@dataclass
class ValidationReport:
    levels_passed: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    n_errors_by_code: dict = field(default_factory=dict)
    score: float = 0.0
    fully_valid: bool = False

    def to_dict(self):
        return asdict(self)

    def messages(self, limit=12):
        return [e["msg"] for e in self.errors[:limit]]


class _R:
    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, code, node, msg):
        self.errors.append({"code": code, "level": CODE_LEVEL[code], "node": node, "msg": msg})

    def warn(self, code, node, msg):
        self.warnings.append({"code": code, "level": CODE_LEVEL.get(code, "V?"), "node": node, "msg": msg})


def _lbl(i, n):
    return f"node {i} '{n.get('name')}'"


def _bcast(shapes):
    out = []
    for dims in zip(*[list(reversed(s)) for s in shapes]):
        vals = {d for d in dims if d != 1}
        if len(vals) > 1:
            return None
        out.append(vals.pop() if vals else 1)
    longest = max(shapes, key=len)
    rest = list(reversed(longest))[len(out):]
    return list(reversed(out + rest))


def _structure(cg, r):
    nodes, edges = cg["nodes"], cg["edges"]
    n = len(nodes)
    seen = {}
    for i, nd in enumerate(nodes):
        if nd["name"] in seen:
            r.err("DUP_NAME", i, f"{_lbl(i, nd)} duplicates the name of node {seen[nd['name']]}")
        else:
            seen[nd["name"]] = i
    good, eset = [], set()
    for a, b in edges:
        if not (0 <= a < n and 0 <= b < n):
            r.err("BAD_EDGE", None, f"edge [{a}, {b}] points outside the {n} nodes")
            continue
        if a == b:
            r.err("SELF_LOOP", a, f"edge [{a}, {b}] is a self-loop on {_lbl(a, nodes[a])}")
            continue
        if (a, b) in eset:
            r.err("DUP_EDGE", b, f"edge [{a}, {b}] is listed twice")
            continue
        eset.add((a, b))
        good.append((a, b))
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(good)
    ph = [i for i, nd in enumerate(nodes) if nd["op"] == "placeholder"]
    outs = [i for i, nd in enumerate(nodes) if nd["op"] == "output"]
    if not ph:
        r.err("NO_INPUT", None, "graph has no placeholder (input) node")
    if len(outs) != 1 or (outs and outs[-1] != n - 1):
        r.err("OUTPUT_COUNT", None, f"graph needs exactly one output node as the last node (found {len(outs)})")
    for i in ph:
        if G.in_degree(i):
            r.err("INPUT_HAS_PRED", i, f"placeholder {_lbl(i, nodes[i])} has incoming edges")
    for i in outs:
        if G.in_degree(i) < 1 or G.out_degree(i) > 0:
            r.err("OUTPUT_DEGREE", i, f"output node needs >=1 input and no outgoing edges "
                                      f"(has {G.in_degree(i)} in, {G.out_degree(i)} out)")
    if not nx.is_directed_acyclic_graph(G):
        cyc = nx.find_cycle(G)
        r.err("CYCLE", cyc[0][0], f"graph has a cycle through nodes {[u for u, _ in cyc]}")
    for a, b in good:
        if a > b:
            r.err("ORDER", b, f"edge [{a}, {b}] goes backwards: nodes must be listed in execution order")
    sources = set(ph) | {i for i, nd in enumerate(nodes) if nd["op"] == "get_attr"}
    reach = set(sources)
    for s in sources:
        reach |= nx.descendants(G, s)
    for i, nd in enumerate(nodes):
        if i not in reach:
            r.err("ORPHAN", i, f"{_lbl(i, nd)} is not reachable from any input")
    if outs:
        co = nx.ancestors(G, outs[-1]) | {outs[-1]}
        for i, nd in enumerate(nodes):
            if i not in co and nd["op"] != "output":
                r.err("DEAD_END", i, f"{_lbl(i, nd)} does not lead to the output (dead end or dead code)")
    for i, nd in enumerate(nodes):
        op, kind = nd["op"], nd["kind"]
        bad = ((op in ("placeholder", "output") and kind != op) or
               (kind in ("placeholder", "output") and kind != op) or
               (op == "call_module" and not kind[:1].isupper()) or
               (op in ("call_function", "call_method") and kind != kind.lower()))
        if bad:
            r.err("OP_KIND", i, f"{_lbl(i, nd)} has op '{op}' but kind '{kind}'")
    return G


def _arity(cg, G, r, info):
    nodes = cg["nodes"]
    for i, nd in enumerate(nodes):
        k, deg = nd["kind"], G.in_degree(i)
        if nd["op"] == "call_function" and k in BINARY_FUNCS:
            if deg == 0:
                r.err("ARITY", i, f"{_lbl(i, nd)} has no inputs; '{k}' needs a tensor input")
            elif deg == 1:
                msg = f"{_lbl(i, nd)} has 1 input; '{k}' of two tensors needs 2 (missing skip edge?)"
                if info is not None and not info.has_scalar_binop:
                    r.err("ARITY", i, msg)
                else:
                    r.warn("ARITY", i, msg)
        elif nd["op"] == "call_function" and k in MULTI_FUNCS and deg < 2:
            r.err("ARITY", i, f"{_lbl(i, nd)} has {deg} input(s); '{k}' joins >=2 tensors (missing branch edge?)")
        elif nd["op"] == "call_function" and k in SINGLE_INPUT_FUNCS and deg != 1:
            r.err("ARITY", i, f"{_lbl(i, nd)} has {deg} inputs; '{k}' takes exactly 1")
        elif nd["op"] == "call_module" and k in SINGLE_INPUT_MODULES and deg != 1:
            r.err("ARITY", i, f"{_lbl(i, nd)} has {deg} inputs; a {k} layer takes exactly 1")
        elif nd["op"] == "output" and deg != 1 and not (info is not None and info.returns_tuple):
            r.err("ARITY", i, f"output node has {deg} inputs; forward() returns a single tensor")


def _module_attr(name, info):
    """Map a predicted call_module name to (flat leaf name, call index) or None."""
    if name in info.leaves:
        return name, 0
    m = re.match(r"^(.*)_(\d+)$", name)
    if m and m.group(1) in info.leaves:
        return m.group(1), int(m.group(2))
    return None


def _code(cg, r, info):
    nodes = cg["nodes"]
    used = {}
    for i, nd in enumerate(nodes):
        if nd["op"] != "call_module":
            continue
        hit = _module_attr(nd["name"], info)
        if hit is None:
            r.err("UNDECLARED_MODULE", i, f"{_lbl(i, nd)} is not a module declared in __init__ (hallucinated?)")
            continue
        leaf = info.leaves[hit[0]]
        used.setdefault(hit[0], []).append(i)
        if leaf.cls != nd["kind"]:
            r.err("KIND_MISMATCH", i, f"{_lbl(i, nd)} has kind '{nd['kind']}' but self.{leaf.path} is a {leaf.cls}")
    for attr, k in info.live_calls.items():
        if attr not in info.leaves:
            continue
        have = len(used.get(attr, []))
        if have < k:
            r.err("MISSING_MODULE", None, f"self.{info.leaves[attr].path} is used {k}x in forward() but appears "
                                          f"{have}x in the graph")
    for attr in info.dead:
        for i in used.get(attr, []):
            r.err("DEAD_CODE_INCLUDED", i, f"{_lbl(i, nodes[i])} is dead code (its result never reaches the return)")


def _expected_input_shape(cg, G, info, ph_index):
    frontier, seen = list(G.successors(ph_index)), set()
    while frontier:
        u = frontier.pop(0)
        if u in seen:
            continue
        seen.add(u)
        nd = cg["nodes"][u]
        if nd["op"] == "call_module":
            hit = _module_attr(nd["name"], info)
            if hit:
                leaf = info.leaves[hit[0]]
                a = leaf.args + [None] * 2
                if leaf.cls == "Linear":
                    f = leaf.kwargs.get("in_features", a[0])
                    return [1, f] if isinstance(f, int) else None
                if leaf.cls == "Conv2d":
                    c = leaf.kwargs.get("in_channels", a[0])
                    return [1, c, 16, 16] if isinstance(c, int) else None
        frontier.extend(sorted(G.successors(u)))
    return None


def _shapes(cg, G, r, info, input_shape):
    nodes = cg["nodes"]

    def shp(i):
        s = nodes[i].get("shape")
        return s if isinstance(s, list) and s else None

    def preds(i):
        return sorted(G.predecessors(i))

    phs = [i for i, nd in enumerate(nodes) if nd["op"] == "placeholder"]
    for j, i in enumerate(phs):
        exp = None
        if input_shape is not None:
            exp = list(input_shape[j]) if (len(phs) > 1 and j < len(input_shape)) else (list(input_shape) if len(phs) == 1 else None)
        elif info is not None:
            exp = _expected_input_shape(cg, G, info, i)
        if exp is not None and shp(i) is not None and shp(i) != exp:
            r.err("SHAPE", i, f"input {_lbl(i, nodes[i])} has shape {shp(i)}, expected {exp}")
    for i, nd in enumerate(nodes):
        s, ps = shp(i), preds(i)
        if s is None:
            continue
        ins = [shp(p) for p in ps]
        k = nd["kind"]
        if nd["op"] == "call_module" and info is not None:
            hit = _module_attr(nd["name"], info)
            leaf = info.leaves[hit[0]] if hit else None
            if leaf is not None and leaf.cls == k:
                a = leaf.args + [None] * 2
                if k in ("Conv2d", "Conv1d", "ConvTranspose2d"):
                    oc = leaf.kwargs.get("out_channels", a[1])
                    if isinstance(oc, int) and (len(s) < 2 or s[1] != oc):
                        r.err("SHAPE", i, f"{_lbl(i, nd)} shape {s}: a {k} with out_channels={oc} gives {oc} channels")
                elif k == "Linear":
                    of = leaf.kwargs.get("out_features", a[1])
                    if isinstance(of, int) and s[-1] != of:
                        r.err("SHAPE", i, f"{_lbl(i, nd)} shape {s}: Linear out_features={of}")
        if len(ins) == 1 and ins[0] is not None:
            if (nd["op"] == "call_module" and k in SHAPE_PRESERVING_MODULES) or \
               (nd["op"] in ("call_function", "call_method") and k in SHAPE_PRESERVING_FUNCS) or nd["op"] == "output":
                if s != ins[0]:
                    r.err("SHAPE", i, f"{_lbl(i, nd)} shape {s} should equal its input's shape {ins[0]}")
            elif k == "flatten" and math.prod(s) != math.prod(ins[0]):
                r.err("SHAPE", i, f"{_lbl(i, nd)} shape {s} has {math.prod(s)} elements, input has {math.prod(ins[0])}")
        if nd["op"] == "call_function" and k in BINARY_FUNCS and len(ins) >= 2 and all(x is not None for x in ins):
            b = _bcast(ins)
            if b is None:
                r.err("SHAPE", i, f"{_lbl(i, nd)} inputs {ins} are not broadcast-compatible")
            elif b != s:
                r.err("SHAPE", i, f"{_lbl(i, nd)} shape {s} should be {b} (broadcast of its inputs)")
        if nd["op"] == "call_function" and k == "cat" and len(ins) >= 2 and all(x is not None for x in ins):
            ok = False
            if all(len(x) == len(s) for x in ins):
                for d in range(len(s)):
                    if s[d] == sum(x[d] for x in ins) and all(x[e] == s[e] for x in ins for e in range(len(s)) if e != d):
                        ok = True
                        break
            if not ok:
                r.err("SHAPE", i, f"{_lbl(i, nd)} shape {s} is not a concatenation of its inputs {ins}")


def validate(cg, source_code=None, input_shape=None, parse_info=None, class_name=None):
    r = _R()
    lv = {k: None for k in LEVEL_CODES}
    if cg is None:
        code = "PARSE" if not (parse_info or {}).get("parse_ok") else "SCHEMA"
        r.err(code, None, (parse_info or {}).get("error") or "output could not be parsed as a JSON graph")
        lv["V0"] = False
        return _finish(r, lv, applicable={"V0"})
    applicable = {"V0", "V1", "V2", "V4"}
    if parse_info and not parse_info.get("schema_ok", True):
        r.err("SCHEMA", None, parse_info.get("error") or "schema violation")
    info = None
    if source_code:
        try:
            info = analyze(source_code, class_name)
            applicable.add("V3")
        except Exception as e:  # noqa: BLE001
            r.warn("UNDECLARED_MODULE", None, f"code analysis failed ({type(e).__name__}); V3 skipped")
    G = _structure(cg, r)
    _arity(cg, G, r, info)
    if info is not None:
        _code(cg, r, info)
    _shapes(cg, G, r, info, input_shape)
    return _finish(r, lv, applicable)


def _finish(r, lv, applicable):
    failed = {e["code"] for e in r.errors}
    total = passed = 0
    for level, codes in LEVEL_CODES.items():
        if level not in applicable:
            continue
        lv[level] = not any(c in failed for c in codes)
        total += len(codes)
        passed += sum(c not in failed for c in codes)
    counts = {}
    for e in r.errors:
        counts[e["code"]] = counts.get(e["code"], 0) + 1
    fully = all(lv[k] for k in ("V0", "V1", "V2", "V4")) and lv["V3"] is not False
    return ValidationReport(levels_passed=lv, errors=r.errors, warnings=r.warnings, n_errors_by_code=counts,
                            score=passed / total if total else 0.0, fully_valid=bool(fully))
