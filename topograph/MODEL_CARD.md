# Model card — TopoGraph LoRA adapters for Qwen3.5

Four LoRA adapters that make Qwen3.5 emit the torch.fx forward-pass dataflow graph of a PyTorch `nn.Module`:

| Adapter | Base model | Training data | Size | Folder |
|---|---|---|---|---|
| `qwen35-9b-topograph-aug-lora` (**recommended**) | `Qwen/Qwen3.5-9B` (snapshot `c2022362…`) | train_ft + 2,809 synthetic | 326 MB | `models/qwen35-9b-topograph-aug-lora/` |
| `qwen35-9b-topograph-lora` | `Qwen/Qwen3.5-9B` | train_ft | 326 MB | `models/qwen35-9b-topograph-lora/` |
| `qwen35-2b-topograph-aug-lora` | `Qwen/Qwen3.5-2B` | train_ft + 2,809 synthetic | 139 MB | `models/qwen35-2b-topograph-aug-lora/` |
| `qwen35-2b-topograph-lora` | `Qwen/Qwen3.5-2B` | train_ft | 139 MB | `models/qwen35-2b-topograph-lora/` |

Rebuild full weights with `scripts/merge_lora.py` (safetensors-level merge; the result has the base model's exact
file layout, so vLLM loads it like the base model).

## Training

- **Method:** supervised fine-tuning, LoRA in bf16 (not QLoRA), loss on completion tokens only (the CG-JSON graph
  and one `<|im_end|>`). Text-only: the vision tower and the multi-token-prediction head get no LoRA and stay frozen.
- **LoRA:** r = 32, alpha = 64, dropout 0.05. Targets: `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
  (full-attention + MLP) and `in_proj_qkv in_proj_z out_proj` (Gated DeltaNet linear-attention layers).
- **Hyperparameters:** 2 epochs, cosine schedule, 3% warmup, effective batch 16 (2B: 8 × 2, 9B: 2 × 8),
  lr 2e-4 (2B) / 1e-4 (9B), `max_length` 2,240 (no truncation), gradient checkpointing, seed 643.
- **Data:** `train_ft` = 3,068 samples (official train split minus the novel-topology holdout and a 5%-of-families
  validation split of 160 samples). The "aug" adapters add 2,809 synthetic modules from `scripts/gen_synthetic.py`
  (random compositions of general PyTorch blocks with random names, labelled by torch.fx; 3,200 generated, kept if
  within the token budget; no structure shared with the novel-topology or OOD-1 sets, checked by
  `scripts/audit_synthetic.py`), 5,877 samples in total. Prompt = fixed system prompt + source code, rendered with
  the Qwen3.5 chat template in non-thinking mode; byte-identical at inference.
- **Hardware:** one NVIDIA L40S (48 GB) per run, Prajna HPC (IIT Bombay). Smoke-test speed: 9B 12.1 s/step, peak
  20.8 GiB; 2B 5.7 s/step, peak 9.1 GiB.

| Adapter | Optimizer steps | Training time | First / last training loss | Final validation loss | Sanity (20 val samples, exact) |
|---|---|---|---|---|---|
| 9B + synthetic | 736 | 3.0 h | 0.050 / 8.0e-6 | 8.3e-7 | 20/20 |
| 9B | 384 | 1.5 h | 0.027 / 2.4e-6 | 1.4e-6 | 20/20 |
| 2B + synthetic | 736 | 1.4 h | 0.106 / 7.9e-5 | 2.8e-6 | 20/20 |
| 2B | 384 | 0.5 h | 0.058 / 2.8e-5 | 2.7e-5 | 20/20 |

The losses are near zero because the targets are deterministic given the code; validation loss says little about
generalisation (it is drawn from the same topology families), which is why the novel-topology and OOD sets exist.
Full logs: `runs/train_*/log_history.csv`, figure `results/figures/F9_loss.png`; arguments in
`models/*-lora/training_args.json`.

## Evaluation

Evaluation sets (never used for training, exemplars or early stopping): official test (1,000), class-anonymised test
(1,000), novel-topology set (564), OOD-1 (40 hand-written modules), OOD-2 (21 modules with operators neither the
dataset nor the synthetic generator uses). vLLM, greedy, JSON-schema constrained decoding. Fully valid % /
exact-match % (full table with CIs: `results/tables/main.md`):

| Adapter | test | test_anon | novel_topo | OOD-1 | OOD-2 | OOD-1 with retry | OOD-2 with retry |
|---|---|---|---|---|---|---|---|
| 9B + synthetic | 100 / 100 | 100 / 100 | 100 / 100 | 95.0 / 97.5 | 76.2 / 76.2 | 97.5 / 97.5 | 90.5 / 76.2 |
| 9B | 99.7 / 99.7 | 99.6 / 99.8 | 100 / 100 | 47.5 / 30.0 | 85.7 / 71.4 | 67.5 / 45.0 | 90.5 / 76.2 |
| 2B + synthetic | 100 / 100 | 99.8 / 100 | 100 / 100 | 72.5 / 70.0 | 71.4 / 52.4 | 72.5 / 70.0 | 66.7 / 52.4 |
| 2B | 93.3 / 93.7 | 93.4 / 93.9 | 64.4 / 64.5 | 5.0 / 2.5 | 28.6 / 19.0 | 7.5 / 2.5 | – |

For comparison: base Qwen3.5-9B 3-shot gets 54.9% valid on test, and Qwen3.5-27B-FP8 3-shot gets 91.0% on test,
80.0% on OOD-1 and 95.2% on OOD-2. Retry = up to 2 rounds of validator feedback (no reference); the adapters
were trained on single-turn data only.

## Intended use and limitations

- Intended for research/teaching on LLM topology reasoning, and for drawing graphs of code torch.fx cannot trace.
- Trained on a small family of synthetic architectures (54 distinct topologies, flat module scopes, 16×16 inputs);
  expect degraded accuracy on larger or more varied code. Predicted shapes follow the dataset's input-shape
  convention (batch 1, 16×16 spatial).
- The most common remaining error is tensor shapes through operators absent from training (for example a stride-2
  convolution before `ConvTranspose2d` in OOD-2): the topology is right, the shapes are not.
- Outputs are not guaranteed correct even when they are valid JSON: check them with `tgraph.validate` or compare to
  torch.fx where the code is traceable.
