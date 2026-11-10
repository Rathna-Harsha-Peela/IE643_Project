#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
# Generic inference job: CONFIGS="cfg1 cfg2 ..." [UNTRACEABLE="cfgA cfgB"]. Idempotent and fail-soft.
source jobs/common.sh infer
set +e
for c in ${CONFIGS}; do
  python scripts/generate.py --config $c --backend vllm ${EVALSETS:+--eval_sets $EVALSETS} || echo "CONFIG_FAILED $c"
done
for c in ${UNTRACEABLE:-}; do
  python scripts/generate.py --config $c --eval_sets untraceable --backend vllm || echo "CONFIG_FAILED $c untraceable"
done
echo "INFER_DONE ${CONFIGS}"
