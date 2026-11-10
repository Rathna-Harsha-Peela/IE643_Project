"""Renders the worked examples for the explainer (PNG graphs + examples.json with code, metrics, validator messages).

    python docs/explainer/make_figs.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from tgraph import visualize as V  # noqa: E402
from tgraph.io import load_eval_set, read_jsonl  # noqa: E402
from tgraph.schema import cg_to_canonical  # noqa: E402

FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

_orig = V._dot_graph


def _dot_hi(title=None):          # print resolution
    d = _orig(title)
    d.attr(dpi="170")
    return d


V._dot_graph = _dot_hi

# (example id, eval set, sample id, [(config, label), ...], title, what it shows)
EXAMPLES = [
    ("ex01", "test", "L0_0803", [("base9b_zs_c", "9B zero-shot"), ("ft9b_c", "9B fine-tuned")],
     "A neuron-level MLP (tier L0)", "zero_vs_ft"),
    ("ex02", "test", None, [("ft9b_c", "9B fine-tuned")], "A linear chain with dead code (tier L1)", "deadcode"),
    ("ex03", "test", "L2_0804", [("base9b_fs_c", "9B 3-shot"), ("ft9b_c", "9B fine-tuned")],
     "A residual network (tier L2): the missing skip edge", "skip"),
    ("ex04", "test", "L3_0804", [("base9b_fs_c", "9B 3-shot")], "Parallel branches joined by concat (tier L3)", "branch"),
    ("ex05", "test", "L4_0800", [("ft2b_c", "2B fine-tuned"), ("ft9b_c", "9B fine-tuned")],
     "A held-out nested topology (tier L4): memorisation vs generalisation", "heldout"),
    ("ex06", "test", "L0_0857", [("base9b_zs_c", "9B zero-shot")], "The zero-shot runaway loop", "loop"),
    ("ex07", "ood", "OOD_resnet_nested_2", [("ft9b_c", "9B fine-tuned"), ("ft9b_aug_c", "9B FT + synthetic")],
     "OOD: nested submodules (dotted scopes)", "nested"),
    ("ex08", "ood", "OOD_modulelist_1", [("ft9b_c", "9B fine-tuned"), ("ft9b_aug_c", "9B FT + synthetic")],
     "OOD: an nn.ModuleList loop", "modulelist"),
    ("ex09", "ood", "OOD_se_1", [("ft9b_c_retry", "9B FT, before/after retry")],
     "OOD: Squeeze-and-Excitation, fixed by verify-and-retry", "retry"),
    ("ex10", "ood2", "OOD2_deconv_2", [("ft9b_aug_c", "9B FT + synthetic"), ("ceil_fs_c", "27B 3-shot")],
     "OOD-2: a transposed-convolution decoder (shape failure)", "shape"),
    ("ex11", "ood2", "OOD2_attn_1", [("ft9b_aug_c", "9B FT + synthetic"), ("ceil_fs_c", "27B 3-shot")],
     "OOD-2: attention with matmul and softmax", "attn"),
    ("ex12", "untraceable", "UNT_data_dependent_if", [("ft9b_c", "9B fine-tuned")],
     "Untraceable code: no torch.fx reference exists", "untraceable"),
]


def pred_row(cfg, es, sid):
    for r in read_jsonl(os.path.join(REPO, "results", "raw", cfg, f"{es}.jsonl")):
        if r["sample_id"] == sid:
            return r
    return None


def met_row(cfg, es, sid):
    p = os.path.join(REPO, "results", "raw", cfg, f"{es}.metrics.jsonl")
    if not os.path.exists(p):
        return None
    for r in read_jsonl(p):
        if r["sample_id"] == sid:
            return r
    return None


def main():
    out = []
    test = {r["sample_id"]: r for r in load_eval_set("test")}
    for ex, es, sid, cfgs, title, kind in EXAMPLES:
        refs = {r["sample_id"]: r for r in load_eval_set(es)}
        if sid is None:   # shortest L1 dead-code sample
            sid = min((s for s, r in test.items() if r["tier"] == "L1" and r["augmentation_type"] == "aug_variant_2"),
                      key=lambda s: len(test[s]["source_code"]))
        ref = refs[sid]
        item = {"id": ex, "eval_set": es, "sample_id": sid, "title": title, "kind": kind,
                "code": ref["source_code"], "tier": ref.get("tier"), "aug": ref.get("augmentation_type"),
                "struct_seen": ref.get("struct_seen"), "category": ref.get("category"), "preds": []}
        if ref.get("target_graph"):
            item["ref_png"] = os.path.basename(V.render(ref["target_graph"], os.path.join(FIG, f"{ex}_ref"),
                                                        title="Reference (torch.fx)", fmt="png"))
            item["ref_nodes"] = len(ref["target_graph"]["nodes"])
        for cfg, label in cfgs:
            r = pred_row(cfg, es, sid)
            m = met_row(cfg, es, sid)
            p = {"config": cfg, "label": label, "finish": r.get("finish_reason"), "n_tokens": r.get("n_output_tokens"),
                 "valid": r["validation"]["fully_valid"], "errors": [e["msg"] for e in r["validation"]["errors"]][:4],
                 "raw_head": (r.get("raw_text") or "")[:300], "raw_tail": (r.get("raw_text") or "")[-220:],
                 "rounds": r.get("rounds_used", 0)}
            if m:
                p["metrics"] = {k: m[k] for k in ("node_f1", "edge_f1", "exact_strict", "isomorphic", "graph_similarity",
                                                  "shape_acc", "pps")}
                p["failures"] = m.get("failures")
            tag = cfg
            if kind == "retry":                     # show the first attempt as well
                ic = r.get("initial_cg")
                if ic:
                    from tgraph.validate import validate
                    p["initial_errors"] = [e["msg"] for e in validate(ic, ref["source_code"]).errors][:4]
                    g0 = cg_to_canonical(ic)
                    p["initial_png"] = os.path.basename(V.render(g0, os.path.join(FIG, f"{ex}_{tag}_initial"),
                                                                 title="First attempt", fmt="png"))
                    if ref.get("target_graph"):
                        p["initial_diff_png"] = os.path.basename(V.render_diff(g0, ref["target_graph"],
                                                                 os.path.join(FIG, f"{ex}_{tag}_initial_diff"), fmt="png", legend=False, title="First attempt vs reference"))
            if r.get("cg") and kind != "loop":
                g = cg_to_canonical(r["cg"])
                p["pred_png"] = os.path.basename(V.render(g, os.path.join(FIG, f"{ex}_{tag}_pred"),
                                                          title=f"Predicted ({label})", fmt="png"))
                if ref.get("target_graph"):
                    p["diff_png"] = os.path.basename(V.render_diff(g, ref["target_graph"],
                                                                   os.path.join(FIG, f"{ex}_{tag}_diff"), fmt="png", legend=False, title=f"{label} vs reference"))
            item["preds"].append(p)
        out.append(item)
        print(ex, sid, [(p["config"], p["valid"]) for p in item["preds"]])
    with open(os.path.join(HERE, "examples.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
