# Loader for the IE643 TopoGraph project.  Usage:  source ~/IE643_Project/tgenv.sh [train|infer]
# Every cache points inside ~/IE643_Project so nothing touches the KV-compression project
# (its conda envs, ~/hf, ~/.triton, pip/conda caches stay untouched).
export TG_ROOT=$HOME/IE643_Project
export TG_REPO=$TG_ROOT/topograph
export CONDA_OVERRIDE_CUDA=12.8                  # /usr/local/cuda is unreadable; conda crashes without this
export CONDA_PKGS_DIRS=$TG_ROOT/cache/conda_pkgs
export PIP_CACHE_DIR=$TG_ROOT/cache/pip
export HF_HOME=$TG_ROOT/hf                       # new downloads (0.8B, 2B, Coder-7B) land here
export TRITON_CACHE_DIR=$TG_ROOT/cache/triton    # never ~/.triton: the KV eval depends on its autotune entries
export TMPDIR=$TG_ROOT/cache/tmp
export XDG_CACHE_HOME=$TG_ROOT/cache/xdg              # torch kernels, matplotlib, ... (~/.cache/torch belongs to the KV jobs)
export VLLM_CACHE_ROOT=$TG_ROOT/cache/vllm            # vLLM torch.compile cache (default ~/.cache/vllm)
export FLASHINFER_WORKSPACE_BASE=$TG_ROOT/cache       # flashinfer JIT cache -> $TG_ROOT/cache/.cache/flashinfer
export TORCHINDUCTOR_CACHE_DIR=$TG_ROOT/cache/inductor
export TORCH_HOME=$TG_ROOT/cache/torch
export MODELS=$TG_ROOT/models                    # symlinks to weights + our merged models
export PYTHONPATH=$TG_REPO${PYTHONPATH:+:$PYTHONPATH}
export TOKENIZERS_PARALLELISM=false
if [ -n "${SLURM_JOB_ID:-}" ]; then                  # compute nodes have no internet
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi
source $HOME/miniforge3/etc/profile.d/conda.sh
[ "${1:-train}" = none ] || conda activate $TG_ROOT/envs/tg-${1:-train}
# infer env: conda-forge ffmpeg/ICU need the env's newer libstdc++; some pip wheel loads the system one (/lib64, too
# old) first, so preload the env's copy.
# vLLM's flashinfer top-k/top-p sampler needs nvcc to JIT-compile (no CUDA toolkit here); use vLLM's torch sampler.
if [ "${1:-train}" = infer ]; then export VLLM_USE_FLASHINFER_SAMPLER=0; fi
if [ "${1:-train}" = infer ]; then export LD_PRELOAD=$TG_ROOT/envs/tg-infer/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}; fi
