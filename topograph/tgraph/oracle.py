"""Reference-graph oracle: torch.fx symbolic trace + dead-code elimination + shape propagation.

Verified in the spec to reproduce all 5,000 dataset graphs byte-exactly (tests/test_oracle.py re-checks it).
"""
import ast

import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp


def _module_classes(src, ns):
    names = [n.name for n in ast.parse(src).body if isinstance(n, ast.ClassDef)]
    return [ns[n] for n in names if isinstance(ns.get(n), type) and issubclass(ns[n], nn.Module)]


def _pick_top_class(classes, class_name=None):
    if class_name:
        return next(c for c in classes if c.__name__ == class_name)
    return classes[-1]          # top-level = last-defined nn.Module (submodule classes come first)


def infer_input_shape(gm):
    """Dataset convention: Linear -> (1, in_features); Conv2d -> (1, C_in, 16, 16)."""
    ph = next(n for n in gm.graph.nodes if n.op == "placeholder")
    frontier = list(ph.users)
    while frontier:             # BFS to the first parametric layer that consumes the input
        u = frontier.pop(0)
        if u.op == "call_module":
            m = gm.get_submodule(u.target)
            if isinstance(m, nn.Linear):
                return (1, m.in_features)
            if isinstance(m, nn.Conv2d):
                return (1, m.in_channels, 16, 16)
        frontier.extend(u.users)
    raise ValueError("Cannot infer input shape; pass input_shape explicitly.")


def extract_reference_graph(src, input_shape=None, class_name=None, canonical_input_name="x"):
    ns = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(compile(src, "<user_code>", "exec"), ns)
    classes = _module_classes(src, ns)
    if not classes:
        raise ValueError("No nn.Module subclass found in the code.")
    model = _pick_top_class(classes, class_name)().eval()
    gm = torch.fx.symbolic_trace(model)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    n_ph = len(placeholders)
    if input_shape is None:
        if n_ph > 1:
            raise ValueError("Multi-input model: pass input_shape as a list of shapes, one per input.")
        input_shape = infer_input_shape(gm)          # single-input models only
    shapes = input_shape if (n_ph > 1) else [input_shape]   # multi-input: pass a list of shapes
    ShapeProp(gm).propagate(*[torch.randn(*s) for s in shapes])
    nodes, ids = [], {}
    for i, n in enumerate(gm.graph.nodes):
        ids[n] = f"node_{i}"
        if n.op == "call_module":
            kind = type(gm.get_submodule(n.target)).__name__
        elif n.op == "call_function":
            kind = getattr(n.target, "__name__", str(n.target))
        elif n.op == "call_method":
            kind = str(n.target)
        else:
            kind = n.op
        tm = n.meta.get("tensor_meta")
        name = canonical_input_name if (n.op == "placeholder" and n_ph == 1 and canonical_input_name) else n.name
        nodes.append({"id": ids[n], "name": name, "op": n.op, "kind": kind,
                      "scope": n.target if n.op == "call_module" else "",
                      "shape": list(tm.shape) if tm is not None and hasattr(tm, "shape") else None})
    edges = [{"src": ids[a], "dst": ids[n]} for n in gm.graph.nodes for a in n.all_input_nodes]
    return {"nodes": nodes, "edges": edges}


_WARMUP = """
class _Warm(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        return torch.relu(self.c(x))
"""


def _serve(req_q, resp_q):
    try:   # first trace pays lazy imports (slow on Lustre); keep it out of the per-call timeout
        extract_reference_graph(_WARMUP)
    except Exception:  # noqa: BLE001
        pass
    resp_q.put(("ready", None))
    while True:
        job = req_q.get()
        if job is None:
            return
        jid, src, input_shape, class_name = job
        try:
            resp_q.put((jid, ("ok", extract_reference_graph(src, input_shape, class_name))))
        except BaseException as e:  # noqa: BLE001 - report anything user code raises
            resp_q.put((jid, ("err", f"{type(e).__name__}: {e}")))


class _OracleWorker:
    """One long-lived 'spawn' subprocess: torch is imported once (slow on Lustre), each call has its own timeout,
    and the worker is killed and restarted lazily if a call hangs. (fork is avoided: a forked child can deadlock
    in torch's CPU thread pool once the parent has used it.)"""

    def __init__(self):
        self.proc, self.jid = None, 0

    def _start(self, startup_timeout):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.req, self.resp = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_serve, args=(self.req, self.resp), daemon=True)
        self.proc.start()
        self.resp.get(timeout=startup_timeout)

    def run(self, src, input_shape, class_name, timeout, startup_timeout=600):
        import queue
        if self.proc is None or not self.proc.is_alive():
            self._start(startup_timeout)
        self.jid += 1
        self.req.put((self.jid, src, input_shape, class_name))
        while True:
            try:
                jid, val = self.resp.get(timeout=timeout)
            except queue.Empty:
                self.proc.terminate()
                self.proc = None
                return ("err", f"Timeout: oracle exceeded {timeout}s")
            if jid == self.jid:
                return val


_WORKER = _OracleWorker()


def safe_reference_graph(src, input_shape=None, class_name=None, timeout=20):
    """Run the oracle in a separate process with a timeout (user code is exec'd).

    Returns (graph, None) or (None, "not traceable: <ErrorType>: <msg>").
    """
    status, val = _WORKER.run(src, input_shape, class_name, timeout)
    return (val, None) if status == "ok" else (None, f"not traceable: {val}")
