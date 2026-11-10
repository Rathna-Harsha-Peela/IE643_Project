#!/bin/bash
#SBATCH --job-name=tg-cpusmoke
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# CPU smoke (spec §11): 0.8B, 16 rows, 3 steps -> merge -> HF generate -> parse -> validate -> metrics -> render.
source jobs/common.sh train
export CUDA_VISIBLE_DEVICES=""
python scripts/train_sft.py --config configs/train_smoke.yaml --max_steps 3 --limit_rows 16 --max_length 1024
python scripts/merge_lora.py --base $MODELS/Qwen3.5-0.8B --adapter models/qwen35-08b-smoke-lora \
    --out $MODELS/qwen35-08b-smoke-merged
python - <<'PY'
import os, sys
sys.path.insert(0, ".")
from scripts.generate import HFBackend, run_set
from tgraph.io import load_eval_set
from tgraph.metrics import compute_metrics
from tgraph.schema import cg_to_canonical
from tgraph import visualize as V
cfg = {"name": "cpu_smoke", "model_dir": os.path.expandvars("$MODELS/qwen35-08b-smoke-merged"), "max_tokens": 64}
row = load_eval_set("test_mini")[0]
out = run_set(HFBackend(cfg, 1), cfg, [row], None)[0]
print("raw:", out["raw_text"][:300])
print("parse:", out["parse_info"], "fully_valid:", out["validation"]["fully_valid"])
if out["cg"]:
    m = compute_metrics(cg_to_canonical(out["cg"]), row["target_graph"])
    print("edge_f1", m["edge_f1"], "graph_similarity", m["graph_similarity"])
    print("render:", V.render(cg_to_canonical(out["cg"]), "results/figures/cpu_smoke_pred"))
print("CPU_SMOKE_DONE")
PY
