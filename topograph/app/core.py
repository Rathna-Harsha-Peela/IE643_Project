"""Core handler for the TopoGraph app (UI-free, so it can be tested directly).

handle(code, model, constrained, retry, shape_override, class_name) -> dict with rendered graphs, metrics,
validity report and JSON. Models:
  "Oracle only"                     - torch.fx reference only
  "Precomputed (<config>)"          - stored predictions looked up by the hash of the source code
  "FT-2B (CPU, HF)" / "FT-9B" / ... - live generation (HF backend on CPU, vLLM on GPU)
"""
import glob
import hashlib
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from tgraph import visualize as V  # noqa: E402
from tgraph.io import PROC_DIR, read_jsonl  # noqa: E402
from tgraph.metrics import compute_metrics  # noqa: E402
from tgraph.oracle import safe_reference_graph  # noqa: E402
from tgraph.parse import parse_output  # noqa: E402
from tgraph.properties import ALL_PROPS  # noqa: E402
from tgraph.schema import cg_to_canonical  # noqa: E402
from tgraph.validate import validate  # noqa: E402

MODELS = os.environ.get("MODELS", os.path.join(os.path.dirname(REPO), "models"))
PRECOMP_CONFIGS = ["ft9b_aug_c", "ft9b_c", "ceil_fs_c", "ft2b_aug_c", "ft2b_c", "base9b_fs_c"]
OUT_DIR = os.path.join(tempfile.gettempdir(), "topograph_app")


def src_hash(code):
    return hashlib.sha1(code.strip().encode()).hexdigest()


# ---- precomputed predictions -----------------------------------------------------------------------------------
_PRE = None


def precomputed():
    """{config: {source_hash: prediction row}} from results/raw (+ app/precomputed.jsonl if packaged)."""
    global _PRE
    if _PRE is not None:
        return _PRE
    _PRE = {}
    srcs = {}
    for f in glob.glob(os.path.join(PROC_DIR, "*.jsonl")):
        if os.path.basename(f) in ("records.jsonl",) or os.path.basename(f).startswith("sft_"):
            continue
        for r in read_jsonl(f):
            if "source_code" in r and "sample_id" in r:
                srcs[(os.path.basename(f)[:-6], r["sample_id"])] = r["source_code"]
    for cfg in PRECOMP_CONFIGS:
        for f in glob.glob(os.path.join(REPO, "results", "raw", cfg, "*.jsonl")):
            if f.endswith(".metrics.jsonl") or ".limit" in f:
                continue
            es = os.path.basename(f)[:-6]
            for r in read_jsonl(f):
                code = srcs.get((es, r["sample_id"]))
                if code:
                    _PRE.setdefault(cfg, {})[src_hash(code)] = r
    pj = os.path.join(REPO, "app", "precomputed.jsonl")
    if os.path.exists(pj):
        for r in read_jsonl(pj):
            _PRE.setdefault(r["config"], {}).setdefault(r["hash"], r)
    return _PRE


# ---- live backends (loaded lazily, once) ---------------------------------------------------------------------
_LIVE = {}


def live_models():
    """Model choices whose files exist."""
    out = {}
    for label, d, adapter in [("FT-9B + synthetic (best)", "qwen35-9b-topograph-aug-merged", None),
                              ("FT-9B", "qwen35-9b-topograph-merged", None),
                              ("FT-2B + synthetic", "qwen35-2b-topograph-aug-merged", None),
                              ("FT-2B", "qwen35-2b-topograph-merged", None),
                              ("FT-2B (adapter)", "Qwen3.5-2B", os.path.join(REPO, "models", "qwen35-2b-topograph-lora")),
                              ("Base-9B (few-shot)", "Qwen3.5-9B", None)]:
        path = os.path.join(MODELS, d)
        if os.path.isdir(path) and (adapter is None or os.path.isdir(adapter)):
            out[label] = {"model_dir": path, "adapter": adapter, "few_shot": label.startswith("Base")}
    return out


def _backend(label, mode):
    key = (label, mode)
    if key not in _LIVE:
        from scripts.generate import HFBackend, VLLMBackend
        spec = live_models()[label]
        # APP_GPU_UTIL: fraction of the GPU per loaded model (0.45 lets two 9B-size models share one 48 GB GPU)
        cfg = {"name": label, "model_dir": spec["model_dir"], "adapter": spec["adapter"], "max_tokens": 3072,
               "gpu_memory_utilization": float(os.environ.get("APP_GPU_UTIL", "0.85")), "tensor_parallel_size": 2,
               "dtype": "half", "enforce_eager": True, "max_num_seqs": 4}
        _LIVE[key] = (VLLMBackend(cfg) if mode == "gpu" else HFBackend(cfg, batch_size=1), cfg, spec)
    return _LIVE[key]


def generate_live(code, label, constrained, retry, shape_hint, class_name, mode):
    from scripts.generate import few_shot_exemplars, run_set
    be, cfg, spec = _backend(label, mode)
    cfg = dict(cfg, constrained=constrained and mode == "gpu", retry_rounds=2 if retry else 0)
    row = {"sample_id": "app", "source_code": code, "input_shape_hint": shape_hint,
           "input_shape_override": None, "class_name": class_name or None}
    return run_set(be, cfg, [row], few_shot_exemplars() if spec["few_shot"] else None)[0]


# ---- main handler -------------------------------------------------------------------------------------------------
def parse_shape(s):
    s = (s or "").strip()
    if not s:
        return None
    if ";" in s:
        return [tuple(int(v) for v in part.split(",")) for part in s.split(";")]
    return tuple(int(v) for v in s.split(","))


def handle(code, model="Oracle only", constrained=True, retry=True, shape_override="", class_name="",
           mode="precomputed", expand_neurons=False):
    res = {"error": None, "ref_msg": "", "pred_img": None, "ref_img": None, "diff_img": None, "metrics": None,
           "properties": None, "validity": None, "pred_json": None, "ref_json": None, "notes": []}
    code = (code or "").strip("\n")
    if not code.strip():
        res["error"] = "Please paste or upload some PyTorch code."
        return res
    try:
        compile(code, "<user_code>", "exec")
    except SyntaxError as e:
        res["error"] = f"The code has a syntax error (line {e.lineno}): {e.msg}"
        return res
    if "nn.Module" not in code and "Module)" not in code:
        res["error"] = "No nn.Module subclass found. Paste a class like `class Net(nn.Module): ...`."
        return res
    try:
        shape = parse_shape(shape_override)
    except ValueError:
        res["error"] = "Input shape override must look like `1,3,32,32` (or `1,3,16,16;1,5` for two inputs)."
        return res
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = src_hash(code + model + str(shape))[:12]
    ref, err = safe_reference_graph(code, input_shape=shape, class_name=class_name or None)
    if ref is None:
        res["ref_msg"] = f"Code not traceable by torch.fx ({err.replace('not traceable: ', '')}): no reference; " \
                         "showing LLM prediction only."
    else:
        res["ref_json"] = json.dumps(ref, indent=1)
        res["ref_img"] = V.render(ref, os.path.join(OUT_DIR, f"{tag}_ref"), title="Reference (torch.fx)",
                                  expand_neurons=expand_neurons)
    pred_row = None
    if model == "Oracle only":
        if ref is None:
            res["notes"].append("Oracle only: nothing to show for untraceable code - pick a model.")
        return res
    if model.startswith("Precomputed"):
        cfg = model[model.index("(") + 1:-1]
        pred_row = precomputed().get(cfg, {}).get(src_hash(code))
        if pred_row is None:
            res["notes"].append(f"No stored prediction from {cfg} for this exact code; choose a live model "
                                "or one of the examples.")
            return res
    else:
        if shape is not None:
            res["notes"].append("Input-shape override was appended to the prompt; the fine-tuned model never saw "
                                "that line in training, so predicted shapes are best effort.")
        hint = shape_override.strip() or None
        pred_row = generate_live(code, model, constrained, retry, hint, class_name, mode)
    cg = pred_row.get("cg")
    if cg is None and pred_row.get("raw_text"):
        cg, _ = parse_output(pred_row["raw_text"])
    rep = validate(cg, code, input_shape=shape, parse_info=pred_row.get("parse_info"), class_name=class_name or None)
    res["validity"] = {"levels": rep.levels_passed, "fully_valid": rep.fully_valid, "score": round(rep.score, 3),
                       "errors": [e["msg"] for e in rep.errors], "warnings": [w["msg"] for w in rep.warnings],
                       "rounds_used": pred_row.get("rounds_used", 0)}
    if cg is None:
        res["error"] = "The model output could not be parsed into a graph."
        res["pred_json"] = pred_row.get("raw_text")
        return res
    pred = cg_to_canonical(cg)
    res["pred_json"] = json.dumps(cg, indent=1)
    res["pred_img"] = V.render(pred, os.path.join(OUT_DIR, f"{tag}_pred"), title=f"Predicted ({model})",
                               expand_neurons=expand_neurons)
    if ref is not None:
        res["diff_img"] = V.render_diff(pred, ref, os.path.join(OUT_DIR, f"{tag}_diff"))
        m = compute_metrics(pred, ref)
        res["metrics"] = {k: round(float(m[k]), 3) for k in ["node_f1", "edge_f1", "edge_f1_aligned", "exact_strict",
                                                            "isomorphic", "graph_similarity", "pps", "shape_acc",
                                                            "reach_f1"]}
        res["properties"] = [{"property": k, "reference": m["ref_props"][k], "predicted": m["pred_props"][k],
                              "match": m["ref_props"][k] == m["pred_props"][k]} for k in ALL_PROPS]
    return res


def examples():
    """~10 picks: 2 per tier from test, 3 OOD, 2 untraceable -> [(label, code)]."""
    out = []
    p = os.path.join(PROC_DIR, "test.jsonl")
    if os.path.exists(p):
        rows = read_jsonl(p)
        for t in ["L0", "L1", "L2", "L3", "L4"]:
            out += [(f"test {r['sample_id']}", r["source_code"]) for r in
                    [r for r in rows if r["tier"] == t and r["augmentation_type"] in ("none", "aug_variant_2")][:2]]
    for name, k in (("ood", 3), ("untraceable", 2)):
        p = os.path.join(PROC_DIR, f"{name}.jsonl")
        if os.path.exists(p):
            rows = read_jsonl(p)
            step = max(1, len(rows) // k)
            out += [(r["sample_id"], r["source_code"]) for r in rows[::step][:k]]
    return out
