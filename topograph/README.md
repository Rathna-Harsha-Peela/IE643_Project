# TopoGraph — Making Large Models Draw Topological Graphs

**IE 643 (IIT Bombay, Autumn 2026) · Team MachineUnlearning · Devanshu Jha & Peela Rathna Harsha**

Given the source code of a PyTorch `nn.Module`, a large language model outputs the **forward-pass dataflow graph**
(nodes = layers and operations, edges = tensor flow, with op kind and tensor shape). Predictions are checked by a
validator that does not need the answer, and compared with the reference graph that `torch.fx` produces.

> Every number in this README comes from a script in this repo that actually ran (`results/tables/`, produced by
> `scripts/aggregate.py`); anything not run is marked *not run*. A 31-page explainer with the whole story, code
> walkthrough and 12 worked examples is in [`docs/explainer/explainer.pdf`](docs/explainer/explainer.pdf).

## Pipeline

```
dataset (5k nn.Module sources + torch.fx graphs)
   │  scripts/prepare_data.py: splits, novel-topology holdout, anonymised test, SFT files, token audit
   ▼
fine-tuning (LoRA on Qwen3.5-2B / 9B)                       scripts/train_sft.py, scripts/merge_lora.py
   ▼
inference: vLLM + JSON-schema constrained decoding          scripts/generate.py
           + validator-driven verify-and-retry
   ▼
evaluation + visualisation                                  scripts/run_eval.py, aggregate.py, plots.py,
   validity (V0–V4), node/edge F1, exact match, isomorphism,  gallery.py, failure_analysis.py
   graph similarity (AED), property preservation, per-tier /
   seen-vs-unseen / OOD slices, failure taxonomy, gallery
   ▼
Gradio app (paste code → predicted graph, reference, diff, metrics, validity)      app/app.py
```

## Repository map

| Path | Contents |
|---|---|
| `tgraph/` | the package: `oracle` (torch.fx reference), `schema` (format converters), `prompts`, `parse`, `codeinfo` (AST analysis), `validate`, `align`, `properties`, `metrics`, `failures`, `visualize` |
| `scripts/` | data prep, OOD labelling, training, merging, generation, evaluation, tables, figures, gallery |
| `configs/` | `experiments.yaml` (every inference run), `train_*.yaml`, `graph_schema.json` |
| `jobs/` | SLURM job scripts (GPU smoke, baselines, training, fine-tuned inference, ceiling, CPU eval) |
| `app/` | Gradio interface (`app.py`) and its UI-free handler (`core.py`) |
| `data/ood/` | OOD-1: 40 hand-written out-of-distribution modules + 6 untraceable ones |
| `data/ood2/` | OOD-2: 21 modules written after the synthetic generator was frozen, with operators neither the dataset nor the generator uses |
| `docs/explainer/` | the explainer (LaTeX source, figures, PDF) |
| `data/processed/splits.json` | every split, the held-out structure ids, few-shot exemplar ids |
| `results/` | `tables/`, `figures/`, `figures/gallery/` |
| `tests/` | pytest suite (oracle 5k exact, round-trip, validator on gold + corruptions, metrics, visualiser, app) |
| `DECISIONS.md` | every deviation from the build spec, with the reason |

## Installation

Two conda environments (PyTorch versions differ because vLLM pins its own):

```bash
bash ../setup_envs.sh train   # torch 2.11.0+cu128, transformers 5.x, TRL, PEFT, flash-linear-attention 0.5.2
bash ../setup_envs.sh infer   # torch 2.11.0+cu129, vLLM 0.26.0+cu129, gradio, xgrammar
source ../tgenv.sh train|infer
```

`setup_envs.sh` and `tgenv.sh` live one level up (`~/IE643_Project/`); copies are in `env/` (and at the top level of the submission zip), so copy them one level up first. Both assume the project sits at `~/IE643_Project/topograph`. `tgenv.sh` keeps every cache (conda, pip,
HF, Triton, vLLM, TMPDIR) inside the project folder and sets `$MODELS` (base weights and merged models).

CUDA note: the cluster driver (570) supports CUDA ≤ 12.8. vLLM ≥ 0.27 needs torch 2.13 (no CUDA-12.8 build), so
vLLM 0.26.0 (+cu129, CUDA minor-version compatible) is used. The infer env also needs `torchcodec==0.15.0+cu129`,
conda-forge `ffmpeg<8`, and `LD_PRELOAD` of the env's `libstdc++` (set by `tgenv.sh infer`). Exact versions:
`requirements-*.lock.txt`. The Graphviz `dot` binary comes from conda-forge `graphviz`; without it the visualiser
falls back to matplotlib.

## Model weights

The four LoRA adapters are too large for git (125 MB and 321 MB files), so they are attached to the GitHub release
[`v1.0`](../../releases/tag/v1.0) as `.tar` files. Unpack them into `models/`:

```bash
mkdir -p models && cd models
for a in qwen35-9b-topograph-aug-lora qwen35-9b-topograph-lora qwen35-2b-topograph-aug-lora qwen35-2b-topograph-lora; do
  tar xf ~/Downloads/$a.tar        # downloaded from the release page (private repo: download while logged in)
done
```

Rebuild a merged model (what vLLM loads) on top of the base model from the Hugging Face Hub:
`python scripts/merge_lora.py --base $MODELS/Qwen3.5-9B --adapter models/qwen35-9b-topograph-aug-lora --out $MODELS/qwen35-9b-topograph-aug-merged`.
`qwen35-9b-topograph-aug-lora` (FT-9B + synthetic) is the recommended one; see `MODEL_CARD.md`.

## Data setup

Put `IE643_Topology_Dataset_5k.zip` in `data/raw/` and unzip it to `data/raw/topo5k/` (folders `L0/ … L4/`). Then:

```bash
python scripts/prepare_data.py --tokenizer $MODELS/Qwen3.5-9B
python scripts/build_ood.py            # OOD-1 labels (torch.fx)
python scripts/build_ood.py ood2       # OOD-2 labels
pytest -q                              # 28 tests
```

Synthetic training data for the "+ synthetic" models (3,200 generated, 2,809 kept after the token budget and the
held-out-structure filter, no structure shared with the novel-topology or OOD sets):

```bash
python scripts/gen_synthetic.py --n 3200 --seed 6430
python scripts/build_sft_aug.py --synthetic data/processed/synthetic.jsonl --tokenizer $MODELS/Qwen3.5-9B
python scripts/audit_synthetic.py      # leakage audit
```

## Reproducing

```bash
S="sbatch -p l40 --qos=l40"
$S jobs/gpu_smoke.sh                     # env/CUDA checks + training speed
$S jobs/infer_baselines.sh               # zero-/few-shot baselines (9B, 2B, Coder-7B)
$S jobs/train.sh                         # 2B then 9B LoRA + merge
$S jobs/infer_finetuned.sh               # fine-tuned models, constrained / unconstrained / retry
$S jobs/infer_ceiling.sh                 # Qwen3.5-27B-FP8 ceiling
# "+ synthetic" models: one training + merge + inference job each
CFG=configs/train_9b_aug.yaml BASE=Qwen3.5-9B TAG=qwen35-9b-topograph-aug INFER="ft9b_aug_c ft9b_aug_c_retry" $S --export=ALL jobs/train_one.sh
CFG=configs/train_2b_aug.yaml BASE=Qwen3.5-2B TAG=qwen35-2b-topograph-aug INFER="ft2b_aug_c ft2b_aug_c_retry" $S --export=ALL jobs/train_one.sh
# any other configuration from configs/experiments.yaml
CONFIGS="ft9b_c_retry ceil_zs_c_retry" $S --export=ALL jobs/infer.sh
# CPU: metrics for every prediction file, GED check, tables, figures, failure analysis, galleries, app cache
srun -p interactive --qos=interactive -c 16 --mem=64G -t 4:00:00 bash jobs/cpu_eval.sh
```

On Prajna, `--qos` must equal the partition name, and only the `interactive` partition accepts `srun`. Compute nodes
have no internet; `jobs/common.sh` sets the offline flags.

## The app

```bash
scripts/run_app.sh --mode precomputed  # no GPU: live torch.fx reference + stored predictions for every dataset/OOD sample
scripts/run_app.sh --mode cpu          # + live generation with the Hugging Face backend (slow; no constrained decoding)
scripts/run_app.sh --mode gpu --preload  # + live generation with vLLM (the real demo)
```

**Live GPU demo on Prajna.** From a terminal on the login node (inside `tmux`, so it survives a dropped connection):

```bash
srun -p interactive --qos=interactive --gres=gpu:1 -c 8 --mem=64G -t 2:00:00 --exclude=cn22-a40 bash jobs/app_gpu.sh
```

Wait for `TopoGraph app ready` (about 5 minutes: model load plus one warm-up request), then forward the port the
log prints: in VS Code, Ports → Forward a Port → `<node>:7860`; or from a laptop,
`ssh -L 7860:<node>:7860 23b1835@login1.prajna.iitb.ac.in`, and open http://localhost:7860. Once warm, a graph takes
about 9 s with FT-9B + synthetic. To book a GPU ahead of time instead:
`sbatch -p l40 --qos=l40 -c 8 --mem=64G -t 6:00:00 jobs/app_gpu.sh`.

**Outside Prajna.** Precomputed mode runs on any laptop. FT-2B (+ synthetic) runs in CPU mode but is slow: measured
on 16 cores of a Prajna server, about 2.5 minutes per graph once loaded (4.4 minutes for the first, including the
model load), because Qwen3.5's linear-attention layers fall back to plain PyTorch without a GPU. That is too slow for a
live demo; a GPU (e.g. a Colab T4, untested) is needed. FT-9B needs about 19 GB of GPU memory in bf16 (Colab L4 is tight, A100 is comfortable).
Ship the LoRA adapter (139 MB / 326 MB) and rebuild the merged model with `scripts/merge_lora.py` on top of the base
model from the Hugging Face Hub, or load base + adapter with PEFT directly (checked: identical graph to the merged
model). Apart from the CPU timing above, no outside-Prajna setup has been tested.

The app runs the code you paste (inside a separate process with a 20 s timeout) to build the torch.fx reference.
It is a local demo tool: do not expose it to untrusted users.

## Results

Fully valid % (the prediction passes all five validator levels V0–V4) on each evaluation set; for OOD-1 and OOD-2
the exact-match % (the predicted graph equals the torch.fx graph, names included) is given after the slash. All
runs use greedy decoding with JSON-schema constrained decoding; "retry" = up to 2 rounds of validator feedback
(never the reference). 95% bootstrap CIs and every other metric: `results/tables/main.md`.

| Model / setting | test (1,000) | test_anon (1,000) | novel_topo (564) | OOD-1 (40) | OOD-2 (21) |
|---|---|---|---|---|---|
| Qwen3.5-2B, 3-shot | 1.1 | – | 1.1 | 0.0 / 0.0 | 0.0 / 4.8 |
| Qwen2.5-Coder-7B, 3-shot | 15.4 | – | 6.9 | 10.0 / 5.0 | 47.6 / 33.3 |
| Qwen3.5-9B, zero-shot | 19.9 | – | 10.3 | 12.5 / 5.0 | 38.1 / 28.6 |
| Qwen3.5-9B, 3-shot | 54.9 | 47.0 | 32.3 | 27.5 / 30.0 | 47.6 / 38.1 |
| Qwen3.5-9B, 3-shot + retry | 67.0\* | – | – | 45.0 / 40.0 | 71.4 / 66.7 |
| Qwen3.5-27B-FP8, zero-shot + retry | 88.0\* | – | 87.9 | **100.0** / 87.5 | 90.5 / 85.7 |
| Qwen3.5-27B-FP8, 3-shot | 91.0 | – | 92.4 | 80.0 / 70.0 | **95.2** / **90.5** |
| FT-2B | 93.3 | 93.4 | 64.4 | 5.0 / 2.5 | 28.6 / 19.0 |
| FT-2B + synthetic | **100.0** | 99.8 | **100.0** | 72.5 / 70.0 | 71.4 / 52.4 |
| FT-9B | 99.7 | 99.6 | **100.0** | 47.5 / 30.0 | 85.7 / 71.4 |
| FT-9B + retry | 99.7 | – | **100.0** | 67.5 / 45.0 | 90.5 / 76.2 |
| FT-9B + synthetic | **100.0** | **100.0** | **100.0** | 95.0 / **97.5** | 76.2 / 76.2 |
| FT-9B + synthetic + retry | **100.0** | – | **100.0** | 97.5 / **97.5** | 90.5 / 76.2 |

\* on `test_mini` (a fixed 100-sample subset of test). "–" = not run. The torch.fx oracle scores 100% on every
traceable set by construction (it produced the labels).

**Key findings**

- **Fine-tuning solves the dataset's own distribution** (FT-9B 99.7% on test, 100% on unseen topologies), but the
  plain fine-tuned models collapse on hand-written OOD code (FT-9B 47.5% valid / 30% exact; FT-2B 5%).
- **Memorisation depends on size:** FT-2B reaches 93% on test but only 64% on held-out topologies; FT-9B gets 100%.
- **Synthetic data fixes generalisation:** adding 2,809 generated modules lifts FT-9B on OOD-1 from 30% to 97.5%
  exact and FT-2B from 5% to 72.5% valid, without any structure overlap with the evaluation sets.
- **On constructs nobody trained on (OOD-2) the larger general model wins:** 27B 3-shot 95.2% valid; FT-9B +
  synthetic needs retry to reach 90.5%, and its remaining errors are mostly tensor shapes through unseen operators.
- **Verify-and-retry helps large models only:** FT-9B OOD-1 47.5 → 67.5%, 27B zero-shot OOD-1 67.5 → 100%;
  FT-2B gains nothing (5.0 → 7.5%).
- **Constrained decoding** guarantees parseable JSON but does not improve correctness (9B 3-shot: 54.9% constrained
  vs 54.6% unconstrained). **Self-consistency** (5 samples) adds 2.5 points on OOD-1. **Thinking mode** gets the
  topology right more often (70% exact on OOD-1 at 30k tokens) but rarely finishes in valid format; forcing the
  format after `</think>` reaches 42% valid on test_mini at about 22k tokens per answer.
- **The metric is sound:** our fast aligned edit distance correlates with the true graph edit distance at
  Spearman 0.998 over 1,378 pairs (F8).

<p align="center">
<img src="results/figures/F1_headline.png" width="49%"> <img src="results/figures/F4_seen_vs_unseen.png" width="49%">
<img src="results/figures/F5_robustness.png" width="49%"> <img src="results/figures/F6_validity.png" width="49%">
</p>

More figures (by tier, by size, failures, properties, AED vs GED, training loss) are in `results/figures/`; galleries of
predicted vs reference graphs are in `results/figures/gallery/`.

## Limitations

- The dataset has only 54 distinct topologies (L4 has 4), so the official test measures unseen *code* with seen
  *topology*; the novel-topology split and the OOD set test generalisation.
- Class names leak the tier (`TierL2_Residual_17`); `test_anon` renames every class to `Net`.
- Hierarchy in the labels is flat; the OOD set adds nested submodules with dotted scopes.
- The torch.fx oracle only covers statically traceable code; untraceable examples are shown qualitatively only.
- OOD-1 (40) and OOD-2 (21) are small: their 95% CIs span about ±15 points, so differences under ~10 points there
  are not significant. Both were written by us; OOD-1 also shares authors with the synthetic generator (OOD-2 was
  written after the generator was frozen to address this).
- All inputs follow the dataset's shape convention (batch 1, 16×16 spatial); the validator's code-consistency check
  is approximate for Python control flow; tuple outputs (e.g. `chunk`) are not supported by the output format.
- Single runs with greedy decoding and one seed; no variance over training seeds.

## Team

Devanshu Jha and Peela Rathna Harsha (IE 643, IIT Bombay, Autumn 2026). Built with the help of Claude Code. Compute:
IIT Bombay Prajna HPC (NVIDIA L40S). Models: Qwen3.5 (Alibaba Qwen team), Qwen2.5-Coder.
