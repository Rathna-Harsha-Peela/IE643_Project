"""Synthetic training modules with new topologies, labelled by the torch.fx oracle (night experiment "aug").

    python scripts/gen_synthetic.py --n 3000 --seed 6430

Random architectures composed from general PyTorch building blocks (conv/BN/activations, residual, branch-concat,
pooling, upsample-skip, dense concat, squeeze-excite gating, nested block classes, Sequential, ModuleList loops,
two-input fusion, MLPs with LayerNorm/GELU/Dropout, dead code), random attribute/argument/class names and shuffled
__init__ order. Kept only if: the oracle traces it, the validator gives zero errors on the gold graph, it has
<= 40 nodes, fits the token budget, and its unlabelled structure is NOT one of the novel-topology holdout or OOD
structures (evaluation integrity). Writes data/processed/synthetic.jsonl.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prepare_data import struct_ids  # noqa: E402
from tgraph.io import PROC_DIR, load_eval_set, read_jsonl, write_jsonl  # noqa: E402
from tgraph.oracle import extract_reference_graph  # noqa: E402
from tgraph.schema import canonical_to_cg  # noqa: E402
from tgraph.validate import validate  # noqa: E402

CH = [4, 6, 8, 12, 16, 24, 32]
ATTR_WORDS = ["conv", "c", "layer", "blk", "enc", "dec", "proj", "stage", "branch", "path", "mix", "fuse", "body",
              "unit", "feat", "lin", "fc", "dense", "head", "stem", "down", "up", "gate", "norm", "bn", "act", "pool"]
ARG_NAMES = ["x", "inp", "inputs", "tensor", "feats", "img", "data", "z", "h", "sample"]
CLS_NAMES = ["Net", "Model", "Encoder", "Backbone", "Block", "Classifier", "FeatureNet", "Tower", "Module", "Arch"]


class B:
    """Code builder for one nn.Module class."""

    def __init__(self, rng):
        self.rng, self.init, self.fwd, self.used, self.nv = rng, [], [], set(), 0

    def attr(self, hint):
        for _ in range(100):
            w = hint if self.rng.random() < 0.6 else self.rng.choice(ATTR_WORDS)
            name = f"{w}{self.rng.randint(0, 9)}" if self.rng.random() < 0.5 else f"{w}_{self.rng.choice('abcdefgh')}{self.rng.randint(0, 9)}"
            if name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("names")

    def var(self):
        self.nv += 1
        return f"{self.rng.choice(['h', 'y', 'o', 't', 'f', 'a', 'b', 'u'])}{self.nv}"

    def mod(self, hint, ctor):
        n = self.attr(hint)
        self.init.append(f"self.{n} = {ctor}")
        return n

    def line(self, s):
        self.fwd.append(s)


def act(b, v):
    r = b.rng.random()
    if r < 0.35:
        return f"torch.relu({v})"
    if r < 0.5:
        return f"F.relu({v})"
    if r < 0.6:
        return f"torch.sigmoid({v})"
    if r < 0.67:
        return f"torch.tanh({v})"
    cls = b.rng.choice(["nn.ReLU()", "nn.GELU()", "nn.SiLU()", "nn.Sigmoid()", "nn.Tanh()", "nn.LeakyReLU(0.1)"])
    return f"self.{b.mod('act', cls)}({v})"


def conv_unit(b, v, C, H, cout=None, allow_stride=True):
    cout = cout or b.rng.choice(CH)
    k = b.rng.choice([1, 3, 3])
    stride = 2 if (allow_stride and H >= 8 and b.rng.random() < 0.15) else 1
    pad = k // 2
    extra = f", stride={stride}" if stride > 1 else ""
    c = b.mod("conv", f"nn.Conv2d({C}, {cout}, kernel_size={k}, padding={pad}{extra})")
    expr = f"self.{c}({v})"
    if b.rng.random() < 0.5:
        expr = f"self.{b.mod('bn', f'nn.BatchNorm2d({cout})')}({expr})"
    if b.rng.random() < 0.8:
        expr = act(b, expr)
    o = b.var()
    b.line(f"{o} = {expr}")
    return o, cout, H // stride


def residual(b, v, C, H):
    o, c2, H2 = v, C, H
    mid = b.rng.choice(CH)
    for i in range(b.rng.randint(1, 3)):
        last = i == 2 or b.rng.random() < 0.5
        o, c2, H2 = conv_unit(b, o, c2, H2, cout=None if not last else b.rng.choice([C, C, mid]), allow_stride=False)
        if last:
            break
    if c2 != C:
        p = b.mod("proj", f"nn.Conv2d({C}, {c2}, kernel_size=1)")
        skip = f"self.{p}({v})"
    else:
        skip = v
    s = b.var()
    b.line(f"{s} = {act(b, f'{o} + {skip}') if b.rng.random() < 0.7 else f'{o} + {skip}'}")
    return s, c2, H


def branch_cat(b, v, C, H):
    outs, ctot = [], 0
    for _ in range(b.rng.randint(2, 4)):
        if b.rng.random() < 0.2:
            p = b.mod("pool", b.rng.choice(["nn.MaxPool2d(3, stride=1, padding=1)", "nn.AvgPool2d(3, stride=1, padding=1)"]))
            o = b.var()
            b.line(f"{o} = self.{p}({v})")
            c = C
        else:
            o, c = v, C
            for _ in range(b.rng.randint(1, 3)):
                o, c, _ = conv_unit(b, o, c, H, allow_stride=False)
        outs.append(o)
        ctot += c
    o = b.var()
    b.line(f"{o} = torch.cat([{', '.join(outs)}], dim=1)")
    return o, ctot, H


def pool(b, v, C, H):
    if H < 4:
        return v, C, H
    p = b.mod("pool", b.rng.choice(["nn.MaxPool2d(2)", "nn.AvgPool2d(2)", "nn.MaxPool2d(kernel_size=2, stride=2)"]))
    o = b.var()
    b.line(f"{o} = self.{p}({v})")
    return o, C, H // 2


def unet(b, v, C, H):
    if H < 4:
        return residual(b, v, C, H)
    e, ce, _ = conv_unit(b, v, C, H, allow_stride=False)
    d, cd, Hd = pool(b, e, ce, H)
    m, cm, _ = conv_unit(b, d, cd, Hd, allow_stride=False)
    u = b.var()
    if b.rng.random() < 0.5:
        up = b.mod("up", "nn.Upsample(scale_factor=2)")
        b.line(f"{u} = self.{up}({m})")
    else:
        b.line(f"{u} = F.interpolate({m}, scale_factor=2.0)")
    if b.rng.random() < 0.7:
        o = b.var()
        b.line(f"{o} = torch.cat([{u}, {e}], dim=1)")
        return conv_unit(b, o, cm + ce, H, allow_stride=False)
    if cm == ce:
        o = b.var()
        b.line(f"{o} = {u} + {e}")
        return o, ce, H
    o = b.var()
    b.line(f"{o} = torch.cat([{e}, {u}], dim=1)")
    return o, ce + cm, H


def dense(b, v, C, H):
    feats, cs = [v], [C]
    for _ in range(b.rng.randint(2, 3)):
        src = feats[0] if len(feats) == 1 else None
        if src is None:
            src = b.var()
            b.line(f"{src} = torch.cat([{', '.join(feats)}], dim=1)")
        g = b.rng.choice([4, 6, 8])
        o, _, _ = conv_unit(b, src, sum(cs), H, cout=g, allow_stride=False)
        feats.append(o)
        cs.append(g)
    o = b.var()
    b.line(f"{o} = torch.cat([{', '.join(feats)}], dim=1)")
    return o, sum(cs), H


def se(b, v, C, H):
    r = max(2, C // 4)
    s = b.var()
    if b.rng.random() < 0.5:
        b.line(f"{s} = {v}.mean((2, 3))")
        f1 = b.mod("fc", f"nn.Linear({C}, {r})")
        f2 = b.mod("fc", f"nn.Linear({r}, {C})")
        g = b.var()
        b.line(f"{g} = torch.sigmoid(self.{f2}(torch.relu(self.{f1}({s}))))")
        o = b.var()
        b.line(f"{o} = {v} * {g}.unsqueeze(-1).unsqueeze(-1)")
    else:
        p = b.mod("pool", "nn.AdaptiveAvgPool2d(1)")
        c1 = b.mod("gate", f"nn.Conv2d({C}, {r}, kernel_size=1)")
        c2 = b.mod("gate", f"nn.Conv2d({r}, {C}, kernel_size=1)")
        b.line(f"{s} = torch.sigmoid(self.{c2}(F.relu(self.{c1}(self.{p}({v})))))")
        o = b.var()
        b.line(f"{o} = {v} * {s}")
    return o, C, H


def seq_block(b, v, C, H):
    items, c = [], C
    for _ in range(b.rng.randint(1, 3)):
        co = b.rng.choice(CH)
        items.append(f"nn.Conv2d({c}, {co}, kernel_size=3, padding=1)")
        if b.rng.random() < 0.5:
            items.append(f"nn.BatchNorm2d({co})")
        items.append(b.rng.choice(["nn.ReLU()", "nn.GELU()", "nn.SiLU()"]))
        c = co
    s = b.mod("features", "nn.Sequential(" + ", ".join(items) + ")")
    o = b.var()
    b.line(f"{o} = self.{s}({v})")
    return o, c, H


def modlist(b, v, C, H):
    n = b.rng.randint(2, 4)
    ml = b.mod("layers", f"nn.ModuleList([nn.Conv2d({C}, {C}, kernel_size=3, padding=1) for _ in range({n})])")
    resid = b.rng.random() < 0.5
    o = b.var()
    b.line(f"{o} = {v}")
    lv = b.rng.choice(["layer", "m", "blk", "conv"])
    b.line(f"for {lv} in self.{ml}:")
    b.line(f"    {o} = F.relu({lv}({o}))" + (f" + {o}" if resid else ""))
    return o, C, H


def dropout(b, v, C, H):
    d = b.mod("drop", b.rng.choice(["nn.Dropout2d(0.1)", "nn.Dropout(0.2)"]))
    o = b.var()
    b.line(f"{o} = self.{d}({v})")
    return o, C, H


CONV_BLOCKS = [(residual, 3), (branch_cat, 2), (pool, 1.5), (unet, 1), (dense, 1), (se, 1), (seq_block, 1),
               (modlist, 1), (dropout, 0.5), (conv_unit, 2)]


def pick(rng, table):
    tot = sum(w for _, w in table)
    r = rng.random() * tot
    for f, w in table:
        r -= w
        if r <= 0:
            return f
    return table[-1][0]


def conv_head(b, v, C, H):
    r = b.rng.random()
    if r < 0.4:
        p = b.mod("pool", "nn.AdaptiveAvgPool2d(1)")
        o = b.var()
        flat = f"torch.flatten(self.{p}({v}), 1)" if b.rng.random() < 0.5 else f"self.{p}({v}).flatten(1)"
        b.line(f"{o} = {flat}")
        fc = b.mod("head", f"nn.Linear({C}, {b.rng.randint(2, 12)})")
        b.line(f"return self.{fc}({o})")
    elif r < 0.6 and C * H * H <= 2048:
        o = b.var()
        b.line(f"{o} = torch.flatten({v}, 1)")
        fc = b.mod("head", f"nn.Linear({C * H * H}, {b.rng.randint(2, 12)})")
        b.line(f"return self.{fc}({o})")
    elif r < 0.85:
        c = b.mod("head", f"nn.Conv2d({C}, {b.rng.randint(1, 10)}, kernel_size=1)")
        b.line(f"return self.{c}({v})")
    else:
        b.line(f"return {v}")


def first_conv(b, v, cin):
    o, c, H = conv_unit(b, v, cin, 16, allow_stride=False)
    return o, c, H


def mlp_body(b, v, din):
    d = b.rng.choice([8, 12, 16, 24, 32, 48])
    fc = b.mod("fc", f"nn.Linear({din}, {d})")
    o = b.var()
    b.line(f"{o} = {act(b, f'self.{fc}({v})')}")
    for _ in range(b.rng.randint(1, 4)):
        r = b.rng.random()
        if r < 0.25:
            n = b.mod("norm", f"nn.LayerNorm({d})")
            o2 = b.var()
            b.line(f"{o2} = self.{n}({o})")
        elif r < 0.45:
            l1 = b.mod("fc", f"nn.Linear({d}, {d})")
            o2 = b.var()
            b.line(f"{o2} = {act(b, f'self.{l1}({o})')} + {o}")
        elif r < 0.6:
            dr = b.mod("drop", "nn.Dropout(0.1)")
            o2 = b.var()
            b.line(f"{o2} = self.{dr}({o})")
        elif r < 0.75:
            a1 = b.mod("lin", f"nn.Linear({d}, {d})")
            a2 = b.mod("gate", f"nn.Linear({d}, {d})")
            o2 = b.var()
            b.line(f"{o2} = self.{a1}({o}) * torch.sigmoid(self.{a2}({o}))")
        elif r < 0.88:
            d2 = b.rng.choice([4, 8, 16])
            a1 = b.mod("branch", f"nn.Linear({d}, {d2})")
            a2 = b.mod("branch", f"nn.Linear({d}, {d2})")
            o2 = b.var()
            b.line(f"{o2} = torch.cat([{act(b, f'self.{a1}({o})')}, self.{a2}({o})], dim=1)")
            d = 2 * d2
        else:
            d2 = b.rng.choice([8, 16, 32])
            l1 = b.mod("fc", f"nn.Linear({d}, {d2})")
            o2 = b.var()
            b.line(f"{o2} = {act(b, f'self.{l1}({o})')}")
            d = d2
        o = o2
    return o, d


NESTED = '''
class {name}(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.{c1} = nn.Conv2d(c, c, kernel_size=3, padding=1)
{bn1}        self.{c2} = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, {arg}):
        {o} = {act1}
        {o} = self.{c2}({o})
        return {final}
'''


def nested_class(rng, cname):
    c1, c2 = rng.choice(["conv1", "conv_a", "c1", "first"]), rng.choice(["conv2", "conv_b", "c2", "second"])
    use_bn = rng.random() < 0.5
    arg = rng.choice(["x", "inp", "t"])
    inner = f"self.{c1}({arg})"
    if use_bn:
        inner = f"self.bn({inner})"
    act1 = rng.choice([f"torch.relu({inner})", f"F.relu({inner})"])
    final = rng.choice([f"torch.relu(o + {arg})", f"o + {arg}", f"F.relu(o + {arg})"])
    return NESTED.format(name=cname, c1=c1, c2=c2, bn1="        self.bn = nn.BatchNorm2d(c)\n" if use_bn else "",
                         arg=arg, o="o", act1=act1, final=final)


def assemble(b, cname, args):
    init = list(b.init)
    b.rng.shuffle(init)
    lines = [f"class {cname}(nn.Module):", "    def __init__(self):", "        super().__init__()"]
    lines += [f"        {l}" for l in init]
    lines += ["", f"    def forward(self, {', '.join(args)}):"]
    lines += [f"        {l}" for l in b.fwd]
    return "\n".join(lines) + "\n"


def gen_one(rng):
    b = B(rng)
    cname = f"{rng.choice(CLS_NAMES)}{rng.randint(0, 99) if rng.random() < 0.6 else ''}"
    kind = rng.random()
    prefix = ""
    if kind < 0.62:                                   # single-input conv net
        arg = rng.choice(ARG_NAMES)
        cin = rng.choice([1, 3, 4, 8, 12, 16])
        v, C, H = first_conv(b, arg, cin)
        if rng.random() < 0.2:                        # nested block class used 2-3 times
            bname = rng.choice(["ResBlock", "Unit", "BasicBlock", "Cell"]) + str(rng.randint(0, 9))
            prefix = nested_class(rng, bname)
            for _ in range(rng.randint(2, 3)):
                n = b.mod("block", f"{bname}({C})")
                o = b.var()
                b.line(f"{o} = self.{n}({v})")
                v = o
        for _ in range(rng.randint(1, 4)):
            v, C, H = pick(rng, CONV_BLOCKS)(b, v, C, H)
        if rng.random() < 0.15:                       # dead code
            d = b.mod("unused", rng.choice(["nn.Identity()", f"nn.Conv2d({C}, {C}, kernel_size=1)"]))
            b.line(f"_dead = self.{d}({v})")
        conv_head(b, v, C, H)
        args = [arg]
    elif kind < 0.85:                                 # MLP
        arg = rng.choice(ARG_NAMES)
        din = rng.randint(3, 32)
        v, d = mlp_body(b, arg, din)
        fc = b.mod("head", f"nn.Linear({d}, {rng.randint(1, 10)})")
        b.line(f"return self.{fc}({v})")
        args = [arg]
    else:                                             # two inputs
        a1, a2 = rng.sample(ARG_NAMES, 2)
        if rng.random() < 0.6:
            c1, c2 = rng.choice([1, 3, 4]), rng.choice([1, 3, 4, 8])
            v1, C1, H1 = first_conv(b, a1, c1)
            v2, C2, H2 = first_conv(b, a2, c2)
            if C1 == C2 and rng.random() < 0.5:
                v = b.var()
                b.line(f"{v} = {v1} + {v2}" if rng.random() < 0.6 else f"{v} = {v1} * {v2}")
                C = C1
            else:
                v = b.var()
                b.line(f"{v} = torch.cat([{v1}, {v2}], dim=1)")
                C = C1 + C2
            H = 16
            for _ in range(rng.randint(0, 2)):
                v, C, H = pick(rng, CONV_BLOCKS)(b, v, C, H)
            conv_head(b, v, C, H)
        else:
            v1, d1 = mlp_body(b, a1, rng.randint(3, 16))
            v2, d2 = mlp_body(b, a2, rng.randint(3, 16))
            v = b.var()
            b.line(f"{v} = torch.cat([{v1}, {v2}], dim=1)")
            fc = b.mod("head", f"nn.Linear({d1 + d2}, {rng.randint(1, 10)})")
            b.line(f"return self.{fc}({v})")
        args = [a1, a2]
    return prefix.lstrip("\n") + ("\n\n" if prefix else "") + assemble(b, cname, args), cname, args


def input_shapes(g, src, cname, args):
    if len(args) == 1:
        return None
    # derive each input's shape by the rule-7 convention from the first parametric consumer (via a probe trace)
    import torch
    import torch.nn as nn
    ns = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(compile(src, "<syn>", "exec"), ns)
    gm = torch.fx.symbolic_trace(ns[cname]())
    shapes = []
    for ph in [n for n in gm.graph.nodes if n.op == "placeholder"]:
        frontier, shp = list(ph.users), None
        while frontier and shp is None:
            u = frontier.pop(0)
            if u.op == "call_module":
                m = gm.get_submodule(u.target)
                if isinstance(m, nn.Linear):
                    shp = (1, m.in_features)
                elif isinstance(m, nn.Conv2d):
                    shp = (1, m.in_channels, 16, 16)
            frontier.extend(u.users)
        shapes.append(shp)
    return shapes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=6430)
    ap.add_argument("--max_nodes", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(PROC_DIR, "synthetic.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    splits = json.load(open(os.path.join(PROC_DIR, "splits.json")))
    banned = set(splits["holdout_struct_ids"])
    banned |= {struct_ids(r["target_graph"])[0] for r in load_eval_set("novel_topo") + load_eval_set("ood")}
    banned_kind = {struct_ids(r["target_graph"])[1] for r in load_eval_set("ood")}
    stats = {"tried": 0, "trace_fail": 0, "too_big": 0, "validator": 0, "banned_struct": 0, "dup": 0}
    out, seen_src = [], set()
    while len(out) < args.n:
        stats["tried"] += 1
        try:
            src, cname, fargs = gen_one(rng)
            shp = input_shapes(None, src, cname, fargs)
            g = extract_reference_graph(src, input_shape=shp, class_name=cname)
        except Exception:  # noqa: BLE001
            stats["trace_fail"] += 1
            continue
        if src in seen_src:
            stats["dup"] += 1
            continue
        if len(g["nodes"]) > args.max_nodes or any(n["shape"] is None for n in g["nodes"]):
            stats["too_big"] += 1
            continue
        sid, sidk = struct_ids(g)
        if sid in banned or sidk in banned_kind:
            stats["banned_struct"] += 1
            continue
        rep = validate(canonical_to_cg(g), src, class_name=cname)
        if rep.errors:
            stats["validator"] += 1
            continue
        seen_src.add(src)
        out.append({"sample_id": f"SYN_{len(out):05d}", "tier": "SYN", "split": "train_syn", "augmentation_type": "none",
                    "source_code": src, "target_graph": g, "class_name": cname, "input_shape": shp,
                    "struct_id": sid, "struct_id_kind": sidk, "n_nodes": len(g["nodes"])})
        if len(out) % 250 == 0:
            print(len(out), stats, flush=True)
    write_jsonl(args.out, out)
    ns = [r["n_nodes"] for r in out]
    print("done", stats, "distinct structures:", len({r["struct_id"] for r in out}),
          "nodes min/med/max:", min(ns), sorted(ns)[len(ns) // 2], max(ns))


if __name__ == "__main__":
    main()
