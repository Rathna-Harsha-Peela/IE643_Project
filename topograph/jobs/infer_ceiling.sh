#!/bin/bash
#SBATCH --job-name=tg-ceiling
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# P1: E17 (Coder-7B few-shot) and, if $MODELS/CEILING exists, E10/E11.
source jobs/common.sh infer
python scripts/generate.py --config coder7b_fs_c --backend vllm
if [ -d $MODELS/CEILING ]; then
  for c in ceil_fs_c ceil_zs_c; do python scripts/generate.py --config $c --backend vllm; done
fi
echo "CEILING_DONE"
