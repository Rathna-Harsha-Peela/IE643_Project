#!/bin/bash
#SBATCH --job-name=tg-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
# GPU smoke (spec §3.5): CUDA checks, vLLM + HF 2-sample generation with 0.8B, 20 LoRA steps of 9B for timing.
source jobs/common.sh infer
python - <<'PY'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      "| vllm", vllm.__version__, "| transformers", transformers.__version__)
PY
python scripts/generate.py --config smoke08b_zs --backend vllm --limit 2 --force
python scripts/generate.py --config smoke08b_zs --backend hf --limit 2 --force --out_dir results/raw_hf_smoke
unset LD_PRELOAD; conda activate $TG_ROOT/envs/tg-train
python -c "import torch; print('train env torch', torch.__version__, torch.cuda.is_available())"
python scripts/train_sft.py --config configs/train_9b.yaml --max_steps 20 --timing_only
python scripts/train_sft.py --config configs/train_2b.yaml --max_steps 10 --timing_only
echo "SMOKE_DONE"
