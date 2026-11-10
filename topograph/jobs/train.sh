#!/bin/bash
#SBATCH --job-name=tg-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# 2B then 9B LoRA SFT (resumable), then merge both. WHICH=2b|9b|both (default both).
source jobs/common.sh train
WHICH=${WHICH:-both}
if [ "$WHICH" != 9b ]; then
  [ -f models/qwen35-2b-topograph-lora/adapter_model.safetensors ] || python scripts/train_sft.py --config configs/train_2b.yaml
  [ -d $MODELS/qwen35-2b-topograph-merged ] || python scripts/merge_lora.py --base $MODELS/Qwen3.5-2B \
      --adapter models/qwen35-2b-topograph-lora --out $MODELS/qwen35-2b-topograph-merged
fi
if [ "$WHICH" != 2b ]; then
  [ -f models/qwen35-9b-topograph-lora/adapter_model.safetensors ] || python scripts/train_sft.py --config configs/train_9b.yaml
  [ -d $MODELS/qwen35-9b-topograph-merged ] || python scripts/merge_lora.py --base $MODELS/Qwen3.5-9B \
      --adapter models/qwen35-9b-topograph-lora --out $MODELS/qwen35-9b-topograph-merged
fi
echo "TRAIN_DONE"
