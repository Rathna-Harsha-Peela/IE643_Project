"""Runtime switches that must run before transformers loads Qwen3.5's modeling code."""
import sys


def cpu_disable_fla():
    """transformers picks flash-linear-attention's Triton kernels for Gated DeltaNet whenever `fla` imports, regardless
    of device; Triton has no CPU driver. On a CPU-only run, make `fla` unimportable so the reference PyTorch path is
    used. Returns True if it was disabled."""
    import torch
    if torch.cuda.is_available():
        return False
    sys.modules["fla"] = None          # `import fla` now raises ImportError -> transformers falls back
    return True
