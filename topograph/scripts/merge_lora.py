"""Merge a LoRA adapter into the base weights at the safetensors level (spec §11).

W' = W + (alpha / r) * B @ A for every adapted Linear. All other tensors (vision tower, MTP head, norms) and every
non-weight file (config, tokenizer, chat template, processor files) are copied unchanged, so the merged folder has
exactly the base model's structure and vLLM loads it like the base model.

    python scripts/merge_lora.py --base $MODELS/Qwen3.5-9B --adapter models/qwen35-9b-topograph-lora \
        --out $MODELS/qwen35-9b-topograph-merged
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def lora_pairs(adapter_dir):
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    mods = {}
    for k, v in sd.items():
        for part in ("lora_A", "lora_B"):
            tag = f".{part}."
            if tag in k:
                mods.setdefault(k.split(tag)[0], {})[part] = v
    return mods, scale


def resolve(mod_path, base_keys):
    """'base_model.model.model.layers.3.mlp.up_proj' -> 'model.language_model.layers.3.mlp.up_proj.weight'."""
    p = mod_path.replace("base_model.model.", "", 1)
    tail = p[p.index("layers."):] if "layers." in p else p
    cands = [k for k in base_keys if k.endswith(tail + ".weight") and "visual" not in k and not k.startswith("mtp")]
    if len(cands) != 1:
        raise KeyError(f"cannot map {mod_path}: candidates {cands[:4]}")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, out = os.path.realpath(args.base), args.out
    assert os.path.realpath(out) != base and not os.path.realpath(out).startswith(base + "/"), "never write into the base folder"
    os.makedirs(out, exist_ok=True)
    idx_path = os.path.join(base, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]
    mods, scale = lora_pairs(args.adapter)
    target = {resolve(m, wmap): ab for m, ab in mods.items()}
    print(f"{len(target)} adapted weights, scale {scale}")
    by_shard = {}
    for k, s in wmap.items():
        by_shard.setdefault(s, []).append(k)
    done = 0
    for shard, keys in sorted(by_shard.items()):
        tensors = {}
        with safe_open(os.path.join(base, shard), framework="pt") as f:
            meta = f.metadata()
            for k in keys:
                t = f.get_tensor(k)
                if k in target:
                    A, B = target[k]["lora_A"].float(), target[k]["lora_B"].float()
                    t = (t.float() + scale * (B @ A)).to(t.dtype)
                    done += 1
                tensors[k] = t.contiguous()
        save_file(tensors, os.path.join(out, shard), metadata=meta or {"format": "pt"})
        print("wrote", shard)
    assert done == len(target), (done, len(target))
    for fn in os.listdir(base):                       # every non-weight file
        if not fn.endswith(".safetensors"):
            src = os.path.join(base, fn)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(out, fn))
    print(f"merged {done} tensors -> {out}")


if __name__ == "__main__":
    main()
