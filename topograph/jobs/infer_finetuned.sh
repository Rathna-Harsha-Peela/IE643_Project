#!/bin/bash
#SBATCH --job-name=tg-ftinfer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# P0: E4 E5 E6, P1: E8 (retry), plus untraceable predictions for the gallery/app. Run after train.sh.
source jobs/common.sh infer
set +e
for c in ft2b_c ft9b_c ft9b_u ft9b_c_retry; do
  python scripts/generate.py --config $c --backend vllm || echo "CONFIG_FAILED $c"
done
for c in ft9b_c ft2b_c; do python scripts/generate.py --config $c --eval_sets untraceable --backend vllm || echo "CONFIG_FAILED $c untraceable"; done
echo "FTINFER_DONE"
