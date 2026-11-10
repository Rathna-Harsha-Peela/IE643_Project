"""TopoGraph Gradio app (spec §14).

    python app/app.py --mode precomputed|cpu|gpu [--port 7860]
"""
import argparse
import json
import os
import sys

import gradio as gr
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import core  # noqa: E402

ABOUT = """**TopoGraph: LLMs drawing neural-network topology.** Paste a PyTorch `nn.Module`; a fine-tuned Qwen3.5
predicts the forward-pass dataflow graph that `torch.fx` would produce (nodes = layers/ops, edges = tensor flow,
with op kind and tensor shape). When the code is traceable, the torch.fx reference, a colour-coded diff and
topology metrics are shown; when it is not (data-dependent control flow), only the LLM prediction exists. The
validator checks the prediction without the reference (structure, arity, consistency with the code, shapes).
Note: the code you paste is executed (inside a subprocess with a timeout) to build the reference. See README.md."""


def build(mode):
    models = ["Oracle only"] + [f"Precomputed ({c})" for c in core.PRECOMP_CONFIGS if c in core.precomputed()]
    default = models[1] if len(models) > 1 else models[0]
    if mode in ("cpu", "gpu"):
        live = list(core.live_models())
        models += live
        if live:
            default = live[0]
    ex = core.examples()
    ex_map = dict(ex)

    def run(code, model, constrained, retry, shape, cls, neurons):
        r = core.handle(code, model, constrained, retry, shape, cls, mode=mode, expand_neurons=neurons)
        msg = "\n".join(filter(None, [f"**Error:** {r['error']}" if r["error"] else "", r["ref_msg"],
                                       *[f"_{n}_" for n in r["notes"]]])) or "Done."
        met = pd.DataFrame([r["metrics"]]).T.reset_index().rename(columns={"index": "metric", 0: "value"}) \
            if r["metrics"] else pd.DataFrame()
        props = pd.DataFrame(r["properties"] or [])
        v = r["validity"]
        if v:
            lv = " · ".join(f"{k}: {'pass' if ok else ('n/a' if ok is None else 'FAIL')}" for k, ok in v["levels"].items())
            vtxt = (f"**Fully valid:** {v['fully_valid']} (score {v['score']}) — retry rounds used: {v['rounds_used']}\n\n"
                    f"{lv}\n\n" + ("\n".join(f"- {e}" for e in v["errors"]) or "No errors.") +
                    ("\n\nWarnings:\n" + "\n".join(f"- {w}" for w in v["warnings"]) if v["warnings"] else ""))
        else:
            vtxt = "No prediction."
        files = []
        for name, txt in (("predicted_graph.json", r["pred_json"]), ("reference_graph.json", r["ref_json"])):
            if txt:
                p = os.path.join(core.OUT_DIR, name)
                open(p, "w").write(txt)
                files.append(p)
        return (msg, r["pred_img"], r["ref_img"], r["diff_img"], met, props, vtxt, r["pred_json"] or "",
                r["ref_json"] or "", files)

    def batch(file, model):
        rows = []
        text = open(file if isinstance(file, str) else file.name).read().strip()
        items = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
        for i, it in enumerate(items):
            r = core.handle(it["source_code"], model, True, False, "", "", mode=mode)
            row = {"index": i, "error": r["error"] or r["ref_msg"],
                   "fully_valid": (r["validity"] or {}).get("fully_valid")}
            row.update(r["metrics"] or {})
            rows.append(row)
        df = pd.DataFrame(rows)
        out = os.path.join(core.OUT_DIR, "batch_metrics.csv")
        df.to_csv(out, index=False)
        summary = df.drop(columns=["index", "error"], errors="ignore").mean(numeric_only=True).round(3).to_dict()
        return df, out, f"{len(df)} items. Means: {summary}"

    with gr.Blocks(title="TopoGraph") as demo:
        gr.Markdown("# TopoGraph — LLMs drawing neural-network topology")
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    exd = gr.Dropdown([k for k, _ in ex], label="Examples", value=None)
                    upload = gr.File(label="Upload .py", file_types=[".py"])
                    code = gr.Code(language="python", label="nn.Module source", lines=22)
                    model = gr.Dropdown(models, value=default, label="Model")
                    with gr.Row():
                        constrained = gr.Checkbox(True, label="Constrained decoding")
                        retry = gr.Checkbox(True, label="Verify & retry")
                        neurons = gr.Checkbox(False, label="Expand small Linear layers into neurons")
                    shape = gr.Textbox(label="Input shape override (optional)", placeholder="1,3,32,32")
                    cls = gr.Textbox(label="Top-level class name (optional)")
                    btn = gr.Button("Generate graph", variant="primary")
                with gr.Column(scale=2):
                    status = gr.Markdown()
                    with gr.Tab("Predicted graph"):
                        pred_img = gr.Image(type="filepath", show_label=False)
                    with gr.Tab("Reference (torch.fx)"):
                        ref_img = gr.Image(type="filepath", show_label=False)
                    with gr.Tab("Diff overlay"):
                        diff_img = gr.Image(type="filepath", show_label=False)
                    with gr.Tab("Metrics"):
                        met = gr.Dataframe(label="Metrics")
                        props = gr.Dataframe(label="Topological properties")
                    with gr.Tab("Validity"):
                        valid = gr.Markdown()
                    with gr.Tab("JSON"):
                        pj = gr.Code(language="json", label="Predicted (CG-JSON)")
                        rj = gr.Code(language="json", label="Reference")
                        dl = gr.File(label="Download")
            exd.change(lambda k: ex_map.get(k, ""), exd, code)
            upload.change(lambda f: open(f if isinstance(f, str) else f.name).read() if f else "", upload, code)
            btn.click(run, [code, model, constrained, retry, shape, cls, neurons],
                      [status, pred_img, ref_img, diff_img, met, props, valid, pj, rj, dl])
        with gr.Tab("Batch"):
            bf = gr.File(label="JSONL or JSON list of {source_code[, target_graph]}")
            bm = gr.Dropdown(models, value=models[0], label="Model")
            bb = gr.Button("Run batch")
            bt = gr.Dataframe()
            bo = gr.File(label="metrics CSV")
            bs = gr.Markdown()
            bb.click(batch, [bf, bm], [bt, bo, bs])
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="precomputed", choices=["precomputed", "cpu", "gpu"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--preload", action="store_true", help="load the default live model before serving")
    a = ap.parse_args()
    demo = build(a.mode)
    if a.preload and a.mode in ("cpu", "gpu") and core.live_models():
        label = next(iter(core.live_models()))
        print(f"preloading {label} ...", flush=True)
        core._backend(label, a.mode)
        # one throw-away request: starts the oracle worker and compiles the JSON grammar (first click ~90 s otherwise)
        core.handle("class Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
                    "        self.fc = nn.Linear(4, 2)\n\n    def forward(self, x):\n        return self.fc(x)\n",
                    label, mode=a.mode)
    print(f"TopoGraph app ready on port {a.port} (mode={a.mode})", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=a.port, share=True)
