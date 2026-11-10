#!/bin/bash
# Launch the TopoGraph app.  scripts/run_app.sh --mode gpu|cpu|precomputed [--port 7860]
#   precomputed : stored predictions + live torch.fx reference (any CPU machine, no model needed)
#   cpu         : adds live generation with the HF backend (slow; use the 2B models)
#   gpu         : adds live generation with vLLM (the real demo; needs a GPU node, see jobs/app_gpu.sh)
set -euo pipefail
MODE=precomputed
for ((i=1; i<=$#; i++)); do [[ "${!i}" == "--mode" ]] && { j=$((i+1)); MODE="${!j}"; }; done
ENV=train; [[ "$MODE" == "gpu" ]] && ENV=infer
source $HOME/IE643_Project/tgenv.sh $ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export GRADIO_ANALYTICS_ENABLED=False
cd $TG_REPO
exec python app/app.py "$@"
