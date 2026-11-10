"""AST analysis of nn.Module source (no execution).

analyze(src, class_name=None) -> CodeInfo with
  - leaves:      {flat_name: Leaf(path, cls, args, kwargs)} for every declared leaf module (nested and
                 Sequential/ModuleList children flattened: self.b1.conv -> "b1_conv", path "b1.conv")
  - live_calls:  {flat_name: k} how many times each leaf module is executed on a path to the return value
  - dead:        declared leaves never called on a live path (e.g. unused_op_K)
  - forward_args, returns_tuple, has_scalar_binop
Best-effort for arbitrary code; exact on the dataset (tests/test_validate.py checks all 5,000 gold graphs).
"""
import ast
from dataclasses import dataclass, field

CONTAINERS = {"Sequential", "ModuleList", "ModuleDict"}


@dataclass
class Leaf:
    path: str
    cls: str
    args: list
    kwargs: dict


@dataclass
class CodeInfo:
    top_class: str = ""
    leaves: dict = field(default_factory=dict)
    live_calls: dict = field(default_factory=dict)
    forward_args: list = field(default_factory=list)
    returns_tuple: bool = False
    has_scalar_binop: bool = False

    @property
    def dead(self):
        return sorted(k for k in self.leaves if self.live_calls.get(k, 0) == 0)

    def scope_map(self):
        return {k: v.path for k, v in self.leaves.items()}


def _lit(node):
    try:
        return ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return None


def _call_class(call):
    """nn.Conv2d(...) / torch.nn.Conv2d(...) / Conv2d(...) / MyBlock(...) -> class name."""
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _self_attr_path(node):
    """self.a.b -> 'a.b'; self.a[0] -> 'a.0'; else None."""
    parts = []
    while True:
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            idx = _lit(node.slice)
            if not isinstance(idx, (int, str)):
                return None
            parts.append(str(idx))
            node = node.value
        else:
            break
    if isinstance(node, ast.Name) and node.id == "self" and parts:
        return ".".join(reversed(parts))
    return None


class _ClassModel:
    def __init__(self, cdef, classes):
        self.cdef = cdef
        self.classes = classes            # name -> ClassDef for every class in the source
        self.decls = {}                   # attr path -> node info dict (tree)
        self._parse_init()

    # ---- __init__: declared modules -------------------------------------------------------------------
    def _module_from_expr(self, expr):
        """Return a declaration dict for an expression that builds a module, or None."""
        if isinstance(expr, ast.Call):
            cls = _call_class(expr)
            if cls is None:
                return None
            if cls in ("Sequential", "ModuleList"):
                items = []
                elems = expr.args
                if cls == "ModuleList" and len(expr.args) == 1:
                    elems = self._list_elems(expr.args[0])
                for a in elems:
                    d = self._module_from_expr(a)
                    if d is not None:
                        items.append(d)
                return {"cls": cls, "children": {str(i): d for i, d in enumerate(items)}}
            return {"cls": cls, "args": [_lit(a) for a in expr.args],
                    "kwargs": {k.arg: _lit(k.value) for k in expr.keywords if k.arg}, "children": None}
        return None

    def _list_elems(self, node):
        if isinstance(node, (ast.List, ast.Tuple)):
            return list(node.elts)
        if isinstance(node, ast.ListComp) and len(node.generators) == 1:
            it = node.generators[0].iter
            if isinstance(it, ast.Call) and _call_class(it) == "range":
                n = _lit(it.args[-1]) if len(it.args) == 1 else None
                if isinstance(n, int):
                    return [node.elt] * n
        return []

    def _parse_init(self):
        init = next((f for f in self.cdef.body if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
        if init is None:
            return
        for st in ast.walk(init):
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                path = _self_attr_path(st.targets[0])
                d = self._module_from_expr(st.value)
                if path and d is not None:
                    self.decls[path] = d

    # ---- leaf flattening --------------------------------------------------------------------------------
    def leaves(self, prefix="", depth=0):
        out = {}
        for path, d in self.decls.items():
            out.update(self._leaves_of(prefix + path, d, depth))
        return out

    def _leaves_of(self, path, d, depth):
        if d.get("children") is not None:
            out = {}
            for k, c in d["children"].items():
                out.update(self._leaves_of(f"{path}.{k}", c, depth))
            return out
        if d["cls"] in self.classes and depth < 8:
            sub = _ClassModel(self.classes[d["cls"]], self.classes)
            return sub.leaves(prefix=path + ".", depth=depth + 1)
        return {path.replace(".", "_"): Leaf(path, d["cls"], d.get("args", []), d.get("kwargs", {}))}

    def _decl_at(self, path):
        parts = path.split(".")
        d = self.decls.get(parts[0])
        for p in parts[1:]:
            if d is None or not d.get("children"):
                return None
            d = d["children"].get(p)
        return d

    def _expand_call(self, path, depth):
        """Leaf calls (flat names -> count) triggered by calling self.<path>(...)."""
        d = self._decl_at(path)
        if d is None:
            return {path.replace(".", "_"): 1}            # undeclared attr: still record it
        if d.get("children") is not None:
            out = {}
            if d["cls"] == "Sequential":
                for k in d["children"]:
                    for n, c in self._expand_call(f"{path}.{k}", depth).items():
                        out[n] = out.get(n, 0) + c
            return out
        if d["cls"] in self.classes and depth < 8:
            sub = _ClassModel(self.classes[d["cls"]], self.classes)
            info = sub.forward_info(depth + 1)
            return {f"{path.replace('.', '_')}_{n}": c for n, c in info["calls"].items()}
        return {path.replace(".", "_"): 1}

    # ---- forward: liveness --------------------------------------------------------------------------------
    def forward_info(self, depth=0):
        fwd = next((f for f in self.cdef.body if isinstance(f, ast.FunctionDef) and f.name == "forward"), None)
        res = {"calls": {}, "args": [], "returns_tuple": False, "scalar_binop": False}
        if fwd is None:
            return res
        res["args"] = [a.arg for a in fwd.args.args[1:]]
        loopvars = {}                                    # loop var -> ModuleList path

        def calls_in(node):
            out = {}
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    p = _self_attr_path(sub.func)
                    if p:
                        for n, c in self._expand_call(p, depth).items():
                            out[n] = out.get(n, 0) + c
                    elif isinstance(sub.func, ast.Name) and sub.func.id in loopvars:
                        ml = self._decl_at(loopvars[sub.func.id]) or {}
                        for k in (ml.get("children") or {}):
                            for n, c in self._expand_call(f"{loopvars[sub.func.id]}.{k}", depth).items():
                                out[n] = out.get(n, 0) + c
                if isinstance(sub, ast.BinOp) and (isinstance(sub.left, ast.Constant) or isinstance(sub.right, ast.Constant)):
                    res["scalar_binop"] = True
            return out

        def names(node, ctx):
            return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}

        # flatten to a list of (defs, uses, calls, is_return) units, compound statements treated as one unit
        units = []
        for st in fwd.body:
            if isinstance(st, ast.For):
                p = _self_attr_path(st.iter)
                if p and isinstance(st.target, ast.Name):
                    loopvars[st.target.id] = p
            if isinstance(st, ast.Return):
                if st.value is not None:
                    res["returns_tuple"] = isinstance(st.value, ast.Tuple)
                    units.append((set(), names(st.value, ast.Load), calls_in(st.value), True))
                continue
            defs = names(st, ast.Store)
            if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                defs.add(st.target.id)
            uses = names(st, ast.Load)
            if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                uses.add(st.target.id)
            units.append((defs, uses, calls_in(st), False))

        live, calls = set(), {}
        for defs, uses, ucalls, is_ret in reversed(units):
            if is_ret or (defs & live):
                live -= defs
                live |= uses                              # x = f(x): x is re-added here
                for n, c in ucalls.items():
                    calls[n] = calls.get(n, 0) + c
        res["calls"] = calls
        return res


def analyze(src, class_name=None):
    tree = ast.parse(src)
    classes = {c.name: c for c in tree.body if isinstance(c, ast.ClassDef)}
    if not classes:
        raise ValueError("no class definition found")
    top = classes[class_name] if class_name else [c for c in tree.body if isinstance(c, ast.ClassDef)][-1]
    m = _ClassModel(top, classes)
    fi = m.forward_info()
    return CodeInfo(top_class=top.name, leaves=m.leaves(), live_calls=fi["calls"], forward_args=fi["args"],
                    returns_tuple=fi["returns_tuple"], has_scalar_binop=fi["scalar_binop"])
