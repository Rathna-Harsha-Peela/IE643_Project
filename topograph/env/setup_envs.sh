#!/bin/bash
# Builds the two project envs under ~/IE643_Project/envs.  Usage: bash setup_envs.sh train|infer
WHICH=$1
source $HOME/IE643_Project/tgenv.sh none      # caches + conda function only, no activation
set -eo pipefail
E=$TG_ROOT/envs/tg-$WHICH
COMMON="networkx matplotlib graphviz jsonschema json-repair pydantic scipy pandas pyyaml pytest gradio pillow"

conda create -y -p $E -c conda-forge --override-channels python=3.11 graphviz
conda activate $E
python -m pip install -U pip

if [ "$WHICH" = train ]; then
  # same torch build + fla version that already run Qwen3.5 on this cluster's A40/A100 (driver 570 = CUDA 12.8)
  pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  pip install "transformers>=5.5.3,<5.20" trl peft accelerate datasets tensorboard \
      flash-linear-attention==0.5.2 $COMMON
else
  # vLLM 0.26.0 is the newest release pinned to torch 2.11; no cu128 build exists, cu129 is the closest
  pip install torch==2.11.0+cu129 torchvision==0.26.0+cu129 torchaudio==2.11.0+cu129 \
      --index-url https://download.pytorch.org/whl/cu129
  pip install "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
      --extra-index-url https://download.pytorch.org/whl/cu129
  pip install peft accelerate $COMMON
fi
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
pip freeze > $TG_ROOT/topograph/requirements-$WHICH.lock.txt
echo "BUILD_OK $WHICH"
