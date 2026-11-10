#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# One LoRA training + merge + inference: CFG=configs/train_X.yaml BASE=<model dir name> TAG=<adapter tag> INFER="<configs>"
source jobs/common.sh train
[ -f models/$TAG-lora/adapter_model.safetensors ] || python scripts/train_sft.py --config $CFG
[ -d $MODELS/$TAG-merged ] || python scripts/merge_lora.py --base $MODELS/$BASE --adapter models/$TAG-lora --out $MODELS/$TAG-merged
echo "TRAIN_DONE $TAG"
if [ -n "${INFER:-}" ]; then
  unset LD_PRELOAD; source $HOME/IE643_Project/tgenv.sh infer
  set +e
  for c in $INFER; do python scripts/generate.py --config $c --backend vllm || echo "CONFIG_FAILED $c"; done
  echo "INFER_DONE $INFER"
fi
