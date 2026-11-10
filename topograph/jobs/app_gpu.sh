#!/bin/bash
# Live GPU demo: the app with vLLM on one GPU, reachable from the login node at http://<node>:7860.
#   srun -p interactive --qos=interactive --gres=gpu:1 -w cn41-l40 -c 8 --mem=64G -t 4:00:00 bash jobs/app_gpu.sh
#   (or sbatch -p l40 --qos=l40 --gres=gpu:1 -c 8 --mem=64G -t 6:00:00 jobs/app_gpu.sh, to book it ahead)
#SBATCH -J tg-app
#SBATCH -o logs/tg-app-%j.out
set -euo pipefail
echo "== $(date) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "== open via:  ssh -L 7860:$(hostname -s):${PORT:-7860} 23b1835@login1.prajna.iitb.ac.in   then http://localhost:7860"
exec bash $HOME/IE643_Project/topograph/scripts/run_app.sh --mode gpu --port ${PORT:-7860} --preload
