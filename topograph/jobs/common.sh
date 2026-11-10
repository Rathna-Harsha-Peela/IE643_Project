# Sourced by every job script: env, offline flags, GPU banner. Usage: source jobs/common.sh train|infer
set -euo pipefail
source $HOME/IE643_Project/tgenv.sh "$1"
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
cd $TG_REPO
mkdir -p logs
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no GPU visible"
