#!/bin/bash
#SBATCH --job-name=tg-baselines
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# P0: E1 E2 E7 E3, then P1: E9 (retry) E16 (thinking) E17 (Coder-7B). Idempotent: finished (config, set) files are
# skipped, so a resubmission continues where it stopped. One failing config does not stop the others.
source jobs/common.sh infer
set +e
for c in base9b_zs_c base9b_fs_c base9b_fs_u base2b_fs_c base9b_fs_c_retry base9b_think coder7b_fs_c; do
  python scripts/generate.py --config $c --backend vllm || echo "CONFIG_FAILED $c"
done
python scripts/generate.py --config base9b_fs_c --eval_sets untraceable --backend vllm || echo "CONFIG_FAILED untraceable"
echo "BASELINES_DONE"
