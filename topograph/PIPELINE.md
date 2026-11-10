# PIPELINE.md — Making Large Models Draw Topological Graphs

**Team MachineUnlearning · IE 643 (IIT Bombay, Autumn 2026) · Devanshu Jha & Peela Rathna Harsha**

This is the build spec for **Claude Code**. It describes the whole project end to end: environment, data handling, model, training, inference, validity checking, metrics, visualization, evaluation, the Gradio interface, packaging, and a timeline. Things already checked on the real dataset are marked **[VERIFIED]**. Everything else is a **well-reasoned starting plan, not a script**: §0.1 says which parts are fixed and which you should adapt as you learn more.

---

## 0. Read this first (instructions to Claude Code)

### 0.1 How binding this spec is
**Fixed. Do not change these without asking the user:**
1. **Honesty of results.** Every number in `results/`, the README and the report notes comes from a script in the repo that actually ran. Experiments that did not run are marked `not run`.
2. **Evaluation integrity.**
   - The official test set (1,000 samples) is used unchanged.
   - Nothing from any evaluation set (test, test_anon, novel_topo, ood) is used for training, few-shot exemplars, prompt tuning or early stopping.
   - The novel-topology holdout is decided before training.
3. **The reference graph conventions** (§2.2) and the oracle (§6). They reproduce the dataset labels exactly. If you change the oracle, re-run the 5,000-sample equality test.
4. **Train/inference consistency.** The fine-tuned model sees exactly the same prompt format at training and at test time.
5. **The P0 deliverables** in §1: model, pipeline, validity checks, visualization, evaluation, interface, documentation.
6. **Cluster usage** follows the user's separate Prajna doc and Claude memory, not this file.

**Flexible. Adapt freely and log the reason in `DECISIONS.md`:**
- file and folder names, module boundaries, CLI flags, config format
- hyperparameters, LoRA targets, batch sizes, epochs, prompt wording details, few-shot exemplar choice
- the exact list of metrics, slices, figures and failure categories (add what is insightful, drop what is not)
- which P1/P2 experiments to run, their order, and the timeline in §16
- library and version choices and workarounds, as long as they are recorded

If something in this spec turns out to be wrong once you see the real environment or data, trust the evidence over the spec, fix it, and note it in `DECISIONS.md`. If you find a better idea that serves the research question in §1, you may pursue it as long as the P0 deliverables are not put at risk.

### 0.2 Working style
1. Read this whole file first. Keep a `TODO.md` (phases in §16 are a reasonable skeleton) and a `DECISIONS.md`.
2. **GPU time is scarce.**
   - Build and test everything that does not need a GPU on CPU first: oracle, data prep, validator, metrics, visualizer, app, and an end-to-end smoke test with a tiny model.
   - Prefer few, self-contained, resumable GPU jobs that write everything to disk and fail fast (load the model and generate a couple of samples at the very start).
   - GPU code should load models from local paths and work with `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so it runs whether or not the GPU machine has internet.
3. **Deadline is about one day of wall-clock time.** Priorities are P0 > P1 > P2. If time gets tight, cut P2, then P1, never P0.
4. Use git: commit at sensible milestones and keep data, weights and raw predictions out of the repo.
5. Prefer simple, robust code: plain Python scripts and a small package, JSONL for data, and tests for the pieces that the numbers depend on (oracle, converters, validator, metrics).
6. **Ask the user** when a decision is genuinely theirs (§19), or when their Prajna doc does not cover something you need. Otherwise make a sensible call, keep going, and log it.

---

## 1. Project brief and how each deliverable is covered

**Task.** Given the source code of a PyTorch `nn.Module` (its `__init__` and `forward()`), a large language model must output the **forward-pass dataflow graph**: nodes are layers and operations, edges are tensor flow, and attributes are operator kind, module scope and tensor shape. The graph must be valid, topologically faithful, and renderable, and it is compared against a reference graph from `torch.fx`.

**Scientific framing (from our prep presentation).** `torch.fx` solves the traceable case exactly, so it is both the **label source** and the **upper-bound baseline**. The research question is how reliably LLMs reason about topology as complexity grows (L0→L4), how far fine-tuning helps, whether that help is *memorized templates* or *generalization* (novel topologies, out-of-distribution code), and what constrained decoding and validator feedback add. LLMs also work where tracing fails (data-dependent control flow, incomplete code); the interface demonstrates that qualitatively.

| Required deliverable | Where it is covered in this spec |
|---|---|
| 1. Topology dataset and task definition | Dataset is done (teammate). We add: task definition (§5), dataset audit and stats (§2, §7), extra splits (novel topology, anonymized, OOD) (§7) |
| 2. Large-model graph generation pipeline | Prompting and fine-tuning Qwen3.5 (§11), constrained decoding with vLLM, verify-and-retry (§12) |
| 3. Graph validity and topological consistency | 5-level validator: schema, structure, arity, code consistency, shapes (§8), plus topological property preservation metrics (§9) |
| 4. Graph visualization module | Graphviz renderer, side-by-side and diff overlay, hierarchy clusters, neuron expansion (§10) |
| 5. Quantitative and qualitative evaluation | Node/edge F1, exact match, isomorphism, GED, property preservation, validity rates, per-tier, size, augmentation and novelty breakdowns, failure taxonomy, gallery (§9, §13) |
| Trained or fine-tuned model | LoRA adapters for Qwen3.5-9B (main) and Qwen3.5-2B, plus a merge script (§11, §17) |
| Code, documentation, dependencies | README, MODEL_CARD, requirements files, job scripts (§17) |
| Interface | Gradio app: paste or upload code, get the predicted graph, reference graph, diff, metrics and validity report (§14) |

---

## 2. The dataset — verified facts

The user has these files (see §3.4 for where the data goes):
`IE643_Topology_Dataset_5k.zip` (all 5,000), `train_data.zip` (4,000), `test_data.zip` (1,000), and `topology_dataset_sample.json` (an old 5-sample pilot).

### 2.1 Layout and schema **[VERIFIED]**
- The zips contain the folders `L0/ … L4/`, each holding `L{t}_{NNNN}.json`, one record per file.
- 5k = train ∪ test exactly. Train = `L*_0000…0799`, test = `L*_0800…0999`, with 800/200 per tier. The record contents are byte-identical across the zips.
- Record fields:
  ```
  sample_id, tier, split, architecture_family_id, is_augmented, augmentation_type,
  source_code, topological_cot, target_graph{nodes[{id,name,op,kind,scope,shape}], edges[{src,dst}]}
  ```
- `source_code` is exactly one `nn.Module` class with `__init__` and `forward`, with no imports. Assume `torch`, `nn` and `F` are in scope.
- 1,250 families × 4 variants. The split is **by family**: there is no family overlap between train and test and no exact source duplicates.
- Augmentations, 1,250 of each type:
  - `none`
  - `aug_variant_1`: forward argument renamed (`inputs`, `inp_tensor`, `feat_map`, `hidden_state`)
  - `aug_variant_2`: dead code inserted (`self.unused_op_K = nn.Identity()` and `_dead_val = self.unused_op_K(x)`)
  - `aug_variant_3`: rename + dead code + shuffled `__init__` order
- **All 4 variants of a family share an identical `target_graph`.**
- Node ids are `node_0…node_{n-1}` in torch.fx execution order. Every edge goes from a lower index to a higher one, and there are no duplicate edges, no null shapes and no dangling references.
- `topological_cot` is a short templated 4-step explanation. Its node count includes the placeholder and output nodes. Treat it as optional (P2 ablation only).
- `topology_dataset_sample.json` is an **old pilot**: its `source_code` contains a whole notebook. **Ignore it.**

### 2.2 Label conventions **[VERIFIED: 5000/5000 exact reproduction with the oracle in §6]**
- Graph = `torch.fx.symbolic_trace` → `graph.eliminate_dead_code()` → `ShapeProp`.
- **The placeholder is always named `"x"`, even when the forward argument is renamed.** Graphs are canonical across augmentations.
- Dead modules are **not** in the graph.
- `scope` = module path (`node.target`) for `call_module` nodes, otherwise `""`. **In this dataset all scopes are flat (depth 0)**, so hierarchy only shows up through naming prefixes such as `b0_`, `pa_` and `pb_`.
- Shapes use batch size 1. Conv nets take input `(1, C_in, 16, 16)`, where C_in is the in_channels of the first conv that consumes the input. MLPs (L0) take `(1, in_features)`.
- `kind` = module class name for `call_module` (`Conv2d`, `BatchNorm2d`, `ReLU`, `GELU`, `Dropout2d`, `Linear`, `MaxPool2d`). For `call_function` it is the function `__name__` (`relu`, `sigmoid`, `tanh`, `cat`, `add`, `flatten`). For placeholder and output nodes it equals the op.
- Node names follow fx: the attribute name for modules, and `relu`, `relu_1`, `relu_2`… for repeated functions. Names are globally unique, and `a + b` becomes `add`.

### 2.3 Size statistics **[VERIFIED]**

| Tier | Meaning | Nodes min/median/max | Edges min/median/max | Source chars (median/max) | Input shape(s) |
|---|---|---|---|---|---|
| L0 | neuron-level toy MLP (per-unit `Linear(k,1)`) | 8 / 10 / 14 | 8 / 11 / 16 | 549 / 783 | (1,2),(1,3),(1,4) |
| L1 | linear chain (conv/bn/act/dropout) | 6 / 15 / 26 | 5 / 14 / 25 | 1087 / 2274 | (1,3,16,16) |
| L2 | residual blocks (skip + optional projection) | 9 / 17 / 26 | 9 / 18 / 28 | 1043 / 1712 | (1,{8,16,24,32},16,16) |
| L3 | multi-branch + concat | 8 / 12 / 20 | 8 / 13 / 22 | 754 / 1167 | (1,{12,16,24},16,16) |
| L4 | nested: stem → 2 paths → cat → merge → residual add → head | 15 / 17 / 19 | 16 / 18 / 20 | 1014 / 1217 | (1,{8,16},16,16) |

The full target JSON is at most about 4.2k characters. The compact model format (§5.2) is at most about 1.5k characters. Prompt plus completion should be around 2k tokens at most; **measure it with the tokenizer** (§7).

### 2.4 Weaknesses that the evaluation design must address **[VERIFIED]**
1. **Topology templates repeat.** Across all 5,000 samples there are only about **68 distinct unlabelled topologies**: L0 has 6, L1 21, L2 14, L3 12, and **L4 only 4**. Every official-test topology also appears in train; only L1 activation-kind sequences are partly novel. The official test therefore measures *unseen code, seen topology*. **We add a novel-topology split and an OOD set (§7).**
2. **Class names leak the tier**: `TierL2_Residual_17` and so on. **We add an anonymized test variant** where the class is renamed to `Net`.
3. Hierarchy is flat in the labels. Hierarchy in the visualizer comes from name-prefix clustering (§10), and the OOD set adds genuinely nested submodules with dotted scopes.
4. "Neurons within layers" exist only at L0, where each unit is a `Linear(k,1)`. The visualizer can optionally expand small `Linear` layers into units (§10).

---

## 3. Compute and environment (cluster-agnostic)

For **how** to run anything on Prajna (submitting jobs, storage paths, modules, environment setup, downloads, data transfer, interactive sessions, port forwarding), follow the user's separate Prajna doc and memory. This section only states **what** the project needs.

### 3.1 GPU requirements
- Any single GPU with ≥40 GB is enough for every P0 job: A100 40/80 GB, A40 48 GB or L40 48 GB.
  - Qwen3.5-9B bf16 LoRA needs about **22 GB** at sequence length 2048 and batch size 1. That is Unsloth's figure; measure it in the GPU smoke test.
  - It fits in 40 GB with gradient checkpointing.
  - The P1 ceiling model needs either an A100 80 GB (Qwen3.5-27B bf16, about 54 GB of weights) or any GPU with ≥40 GB (Qwen2.5-Coder-32B-AWQ, about 20 GB) (§3.3).
- The total P0+P1 budget is about 4–5 GPU-hours on an A100, or about 6 on an A40 or L40 (§15). The hybrid linear-attention layers make 9B training slower per token than a dense 7B model. **Measure seconds per step in the smoke test** and recompute the plan (§11).

### 3.2 Python environments
The user **already has Qwen3.5-9B running** on the cluster. Reuse that environment and those weights for inference, as described in the user's Prajna doc, instead of downloading or installing again. Confirm what it is (vLLM version, transformers version, weights path) and record it in `DECISIONS.md`.

vLLM pins its own torch version, so separate training and inference environments are the safe default:
- **`tg-train`:** `torch` + **`transformers>=5`** (Qwen3.5 does not work on older transformers) + recent `trl` and `peft` compatible with transformers v5, plus `datasets accelerate tensorboard networkx matplotlib graphviz jsonschema json-repair pydantic scipy pandas pyyaml pytest`.
  - **Strongly recommended for training speed:** `flash-linear-attention` (`fla`) and `causal-conv1d`, the fast GPU kernels for Qwen3.5's Gated DeltaNet (linear-attention) layers. Without them transformers falls back to a slow pure-PyTorch path. If they fail to install, continue without them and rely on the timing check in §11.
  - Do **not** install bitsandbytes: QLoRA is discouraged for Qwen3.5.
- **`tg-infer`:** based on the user's existing Qwen3.5 inference environment if it has **vLLM ≥ 0.17** (earlier vLLM releases did not support Qwen3.5).
  - **Do not modify the user's working environment.** Clone it (e.g. `conda create --clone`) or create a new one, and add `transformers peft networkx matplotlib graphviz jsonschema json-repair pydantic scipy pandas pyyaml gradio pytest xgrammar` there.
  - Evaluation, plots and the CPU app only need the lightweight packages, so they can also run from a small CPU-only env.
- The Graphviz **`dot` binary** must be on PATH (conda-forge `graphviz`); otherwise the visualizer uses its matplotlib fallback (§10).
- Record `pip freeze` of both envs as `requirements-*.lock.txt`.
- **CUDA compatibility:** recent PyTorch wheels may target CUDA 13.x, which needs a very new NVIDIA driver. If the GPU driver is older, use a CUDA 12.x build of torch and a vLLM release built against it. Verify with a GPU smoke test (`torch.cuda.is_available()`, `import vllm`, 2-sample generation) **before** any real GPU job.

### 3.3 Models (all loaded from local directories; `$MODELS` points to where they live)
| Model | Role | Approx. size |
|---|---|---|
| `Qwen/Qwen3.5-9B` | **P0 main model** (zero-shot, few-shot, LoRA fine-tune); already set up by the user | ~19 GB |
| `Qwen/Qwen3.5-2B` | P0, small model from the same family for the scaling comparison; also the CPU model for the app | ~4–5 GB |
| `Qwen/Qwen3.5-0.8B` | smoke tests only (same architecture, so the smoke test exercises the real code path) | ~2 GB |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | P1, few-shot baseline: an older **code-specialised** model vs a newer general model | ~15 GB |
| Ceiling, P1: `Qwen/Qwen3.5-27B` (A100 80 GB, bf16) **or** `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` (any ≥40 GB GPU) | zero/few-shot upper bound (no API keys); prefer 27B if an 80 GB A100 is available, since it stays in the same family | ~54 GB / ~19 GB |

**Why these models.**
- **Qwen3.5-9B is the primary model** because the user already has it running, which removes the biggest setup risk.
- It is the newest model in the plan: the small Qwen3.5 models were released in March 2026 under Apache-2.0.
- Its official Hugging Face class is `Qwen3_5ForConditionalGeneration`, which means it is natively **multimodal** (it has a vision encoder). We use it **text-only**.
- It has hybrid **thinking/non-thinking** modes. We use **non-thinking** mode for all main runs, and thinking mode only as an ablation (E16).
- Its architecture mixes standard attention with Gated DeltaNet linear-attention layers. That is why it needs transformers v5 and benefits from the extra kernels in §3.2.
- **Qwen3.5-2B and 0.8B** keep the scaling comparison and the smoke test inside one family, so differences come from scale and not from architecture.
- **Qwen2.5-Coder-7B** (the original candidate on prep slide 22) is kept as a cheap few-shot baseline. It answers whether a code-specialised model beats a newer general model.

### 3.4 Data
The dataset (`IE643_Topology_Dataset_5k.zip`) goes to `data/raw/` and is unzipped to `data/raw/topo5k/`. The 5k zip alone is enough because each record has a `split` field. If the file is not present, ask the user to place it there.

### 3.5 GPU job units (`jobs/`)
A suggested grouping of the GPU work into self-contained scripts; regroup if the cluster's limits make something else better. Each script should be resumable or idempotent (skip outputs that already exist). Wrap them in whatever submission mechanism the user's Prajna doc specifies.
- `jobs/gpu_smoke.sh` (~15 min):
  - env and CUDA checks
  - 2-sample generation with Qwen3.5-0.8B through both vLLM and transformers
  - **20 LoRA training steps of Qwen3.5-9B** at the real batch size and sequence length, logging **seconds per step, peak GPU memory, and whether the fast linear-attention kernels were used**. These numbers decide the training plan (§11).
- `jobs/infer_baselines.sh` (~1 h): every zero-shot and few-shot config that needs no training (§15).
- `jobs/train.sh` (~2–3 h): 2B then 9B sequentially, followed by the LoRA merge (§11).
- `jobs/infer_finetuned.sh` (~1–1.5 h): every fine-tuned config, the retry loop, and the anonymized and OOD sets. Runs after `train.sh` succeeds.
- `jobs/infer_ceiling.sh` (~40 min, P1): the ceiling model (27B or 32B-AWQ) zero-shot and few-shot, plus the Qwen2.5-Coder-7B few-shot baseline.
- `jobs/cpu_eval.sh` (CPU only, ~30 min with 16 cores): metrics for every config, aggregation and plots.

Each GPU script should use `set -euo pipefail`, print the GPU name and memory, exports the offline variables (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false`), and writes logs to `logs/`.

---

## 4. Repository layout (suggested)

```
topograph/
├── README.md  MODEL_CARD.md  PIPELINE.md (this file)  TODO.md  DECISIONS.md
├── requirements-train.txt  requirements-infer.txt  (+ *.lock.txt from pip freeze)
├── configs/
│   ├── experiments.yaml        # every inference run: model, adapter, prompt mode, decoding, datasets
│   ├── train_2b.yaml  train_9b.yaml
│   └── graph_schema.json       # JSON schema for model output (§5.2)
├── tgraph/                     # importable package (CPU-only deps except where noted)
│   ├── __init__.py
│   ├── oracle.py               # §6 torch.fx reference extractor
│   ├── schema.py               # compact format <-> canonical converters, JSON schema
│   ├── prompts.py              # system prompt, user template, few-shot exemplars
│   ├── parse.py                # robust extraction of JSON from raw LLM text
│   ├── codeinfo.py             # AST analysis of source: declared modules, ctor args, live attrs
│   ├── validate.py             # §8 validity checker
│   ├── align.py                # node alignment (name match + Hungarian)
│   ├── metrics.py              # §9 metrics
│   ├── properties.py           # topological property extraction
│   ├── visualize.py            # §10 renderer (graphviz + matplotlib fallback)
│   └── io.py                   # jsonl helpers, dataset loading
├── scripts/
│   ├── prepare_data.py         # §7 splits, SFT files, token audit, stats
│   ├── build_ood.py            # §7.2 label the OOD set with the oracle
│   ├── train_sft.py            # §11
│   ├── merge_lora.py
│   ├── generate.py             # §12 inference (vllm | hf backends), retry loop
│   ├── run_eval.py             # §13 metrics over prediction files (multiprocessing)
│   ├── aggregate.py  plots.py  gallery.py  failure_analysis.py
│   └── run_all.sh              # runs the job units in order (wrapped per the Prajna doc)
├── jobs/                       # §3.5 GPU/CPU job units
├── app/app.py                  # §14 Gradio interface
├── data/
│   ├── raw/                    # zips + unzipped (gitignored)
│   ├── processed/              # records.jsonl, splits.json, sft_*.jsonl, eval sets (gitignored except splits.json)
│   └── ood/                    # hand-written OOD modules *.py + meta.json (committed)
├── models/                     # LoRA adapters (gitignored; packaged separately)
├── results/
│   ├── raw/<config>/<eval_set>.jsonl     # predictions + per-sample metrics (gitignored)
│   ├── tables/*.csv *.md
│   └── figures/*.png *.svg  gallery/
├── logs/
└── tests/                      # pytest: oracle, schema roundtrip, validator, metrics, visualize
```

---

## 5. Task definition and output representation

### 5.1 Formal task
Input: the source code *C* of an `nn.Module`. Output: a directed graph *G = (V, E, A)*.
- *V* is the fx nodes in execution order: placeholder(s), operations, and one output.
- *E ⊆ V×V* is tensor dataflow.
- *A* holds `name`, `op ∈ {placeholder, call_module, call_function, call_method, get_attr, output}`, `kind`, and `shape`.
- The reference is *G\* = T(C)*, where *T* is the oracle (§6).
- Prediction is *Ĝ = f_θ(C)*.
- Goal: maximize agreement under topology-aware measures (§9) while guaranteeing validity (§8).

### 5.2 Model I/O format: "compact graph JSON" (CG-JSON)
This is what the LLM emits. It is shorter than the dataset format: node ids are list positions, edges are index pairs, and `scope` and `id` are derived. Converters live in `tgraph/schema.py`:
```json
{"nodes":[{"name":"x","op":"placeholder","kind":"placeholder","shape":[1,3]},
          {"name":"fc1_u0","op":"call_module","kind":"Linear","shape":[1,1]},
          {"name":"relu","op":"call_function","kind":"relu","shape":[1,1]}],
 "edges":[[0,1],[1,2]]}
```
- `canonical_to_cg(target_graph)`: drop `id` and `scope`, and map `node_k` to k.
- `cg_to_canonical(cg)`:
  - `id = f"node_{i}"`.
  - For `call_module`, `scope = name` (or a dotted path if the OOD source shows nesting; resolve with `codeinfo`).
  - Otherwise `scope = ""`.
- Round-trip test: `cg_to_canonical(canonical_to_cg(g)) == g` for all 5,000 dataset graphs. It holds because scopes are flat.

`configs/graph_schema.json`, used for constrained decoding and validation:
```json
{
  "type": "object",
  "properties": {
    "nodes": {"type": "array", "minItems": 2, "items": {
      "type": "object",
      "properties": {
        "name":  {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
        "op":    {"type": "string", "enum": ["placeholder","call_module","call_function","call_method","get_attr","output"]},
        "kind":  {"type": "string"},
        "shape": {"type": "array", "items": {"type": "integer", "minimum": 1}}
      },
      "required": ["name","op","kind","shape"], "additionalProperties": false}},
    "edges": {"type": "array", "items": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 2, "maxItems": 2}}
  },
  "required": ["nodes","edges"], "additionalProperties": false
}
```
Keep `kind` an open string so OOD layers are possible. If the xgrammar backend rejects `pattern` or `minimum`, drop them and log it in `DECISIONS.md`. The validator checks those properties anyway.

**Representation is an experimental variable** (prep slide 9). The P2 ablation uses a plain-text edge list (`name -> name` lines) in zero-shot mode, with a parser to CG-JSON.

### 5.3 Prompts (`tgraph/prompts.py`), used identically for zero-shot, few-shot and fine-tuning

**SYSTEM_PROMPT** (a tested starting point; it encodes the §2.2 conventions, which are fixed. The wording can be improved, but keep it identical between training and inference):
```
You convert PyTorch nn.Module source code into the forward-pass dataflow graph that
torch.fx.symbolic_trace would produce after dead-code elimination.

Rules:
1. Nodes are listed in execution order: one "placeholder" node per forward() input, then one node per
   executed operation, then exactly one final node {"name":"output","op":"output","kind":"output"}.
   If forward() has a single input, name its placeholder "x" regardless of the argument name.
2. self.<attr>(...) on a submodule -> op "call_module", name = attribute path with "." replaced by "_",
   kind = the module class name (e.g. "Conv2d", "BatchNorm2d", "ReLU", "Linear").
3. torch functions and torch.nn.functional calls (torch.relu, torch.cat, torch.flatten, F.relu, ...) and
   Python operators on tensors (a + b -> "add", a * b -> "mul") -> op "call_function",
   kind = the function name ("relu", "cat", "add", ...).
4. Tensor methods (x.view(...), x.flatten(1), x.mean(...)) -> op "call_method", kind = the method name.
5. Node names are globally unique: the first use of a name is plain ("relu"), later ones get suffixes
   "relu_1", "relu_2", ... in execution order (this also applies when a module attribute is called twice).
6. Skip anything that does not contribute to the returned value (dead code). Plain variable aliasing
   (y = x) creates no node.
7. "shape" is the output tensor shape assuming batch size 1. Conv networks receive 16x16 spatial inputs,
   i.e. [1, C_in, 16, 16]; networks whose first layer is Linear receive [1, in_features]. With several
   inputs, each input's shape follows the same rule using the first layer that consumes it.
8. "edges" is a list of [i, j] pairs (indices into "nodes"): the output of node i is a direct input of
   node j, listed in the order node j consumes them. The output node consumes the returned tensor.

Return only the JSON object {"nodes": [...], "edges": [...]} with no explanation.
```
**USER_TEMPLATE:** ``"Source code:\n```python\n{code}\n```"``

**Chat template and thinking mode (Qwen3.5):**
- Render every prompt with `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)`.
- Check that the rendered text either has no `<think>` block or has an empty one pre-filled by the template. Save one rendered prompt to `DECISIONS.md` as the canonical example.
- **The same rendered string** is used for training (§7) and inference (§12). That guarantees the train and test templates match, and a mismatch is the most common cause of degraded fine-tuned output.
- For other model families (Qwen2.5-Coder, the 32B ceiling), use their own chat template without the `enable_thinking` argument.

**Few-shot mode:** prepend 3 fixed exemplars as user/assistant turns: one L0, one L2 and one L4 sample, taken from the **train_ft** split with `augmentation_type=="none"` and seed 643. Log their ids in `DECISIONS.md`. The assistant turns contain gold CG-JSON (compact separators). Few-shot prompts are about 3k tokens (measure them in the token audit); set `max_model_len=8192` for headroom.

**Fine-tuning:** the same system prompt and user template, with the completion = gold CG-JSON serialized via `json.dumps(cg, separators=(",",":"))`. Use the same serialization at inference time so train and test match.

### 5.4 Robust parsing of raw outputs (`tgraph/parse.py`)
1. Strip markdown fences and any `<think>…</think>` block.
2. Take the first balanced `{…}` substring.
3. Try `json.loads`; on failure use `json_repair.loads` and set `repaired=True`. If that fails too, the result is a `parse_fail`.
4. Validate against the schema. Coerce numeric strings to ints if needed and count every coercion.
5. Return `(cg | None, parse_info)`. Unconstrained runs report parse, repair and schema rates; constrained runs report them too, but say in the write-up that 100% parse validity is guaranteed by construction and is *not* a result (prep slide 9).

---

## 6. The oracle (`tgraph/oracle.py`) **[VERIFIED: reproduces 5000/5000 dataset graphs byte-exactly]**

Start from this code; it is verified against all 5,000 labels. The `n_ph` handling for multi-input OOD models was tested separately on a two-input example. If you modify it, re-run the 5,000-sample equality test.
```python
"""Reference-graph oracle: torch.fx symbolic trace + dead-code elimination + shape propagation."""
import ast
import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

def _module_classes(src, ns):
    names = [n.name for n in ast.parse(src).body if isinstance(n, ast.ClassDef)]
    return [ns[n] for n in names if isinstance(ns.get(n), type) and issubclass(ns[n], nn.Module)]

def _pick_top_class(classes, class_name=None):
    if class_name:
        return next(c for c in classes if c.__name__ == class_name)
    return classes[-1]          # top-level = last-defined nn.Module (submodule classes come first)

def infer_input_shape(gm):
    """Dataset convention: Linear -> (1, in_features); Conv2d -> (1, C_in, 16, 16)."""
    ph = next(n for n in gm.graph.nodes if n.op == "placeholder")
    frontier = list(ph.users)
    while frontier:             # BFS to the first parametric layer that consumes the input
        u = frontier.pop(0)
        if u.op == "call_module":
            m = gm.get_submodule(u.target)
            if isinstance(m, nn.Linear):
                return (1, m.in_features)
            if isinstance(m, nn.Conv2d):
                return (1, m.in_channels, 16, 16)
        frontier.extend(u.users)
    raise ValueError("Cannot infer input shape; pass input_shape explicitly.")

def extract_reference_graph(src, input_shape=None, class_name=None, canonical_input_name="x"):
    ns = {"torch": torch, "nn": nn, "F": torch.nn.functional}
    exec(compile(src, "<user_code>", "exec"), ns)
    model = _pick_top_class(_module_classes(src, ns), class_name)().eval()
    gm = torch.fx.symbolic_trace(model)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    n_ph = len(placeholders)
    if input_shape is None:
        if n_ph > 1:
            raise ValueError("Multi-input model: pass input_shape as a list of shapes, one per input.")
        input_shape = infer_input_shape(gm)          # single-input models only
    shapes = input_shape if (n_ph > 1) else [input_shape]   # multi-input: pass a list of shapes
    ShapeProp(gm).propagate(*[torch.randn(*s) for s in shapes])
    nodes, ids = [], {}
    for i, n in enumerate(gm.graph.nodes):
        ids[n] = f"node_{i}"
        if n.op == "call_module":
            kind = type(gm.get_submodule(n.target)).__name__
        elif n.op == "call_function":
            kind = getattr(n.target, "__name__", str(n.target))
        elif n.op == "call_method":
            kind = str(n.target)
        else:
            kind = n.op
        tm = n.meta.get("tensor_meta")
        name = canonical_input_name if (n.op == "placeholder" and n_ph == 1 and canonical_input_name) else n.name
        nodes.append({"id": ids[n], "name": name, "op": n.op, "kind": kind,
                      "scope": n.target if n.op == "call_module" else "",
                      "shape": list(tm.shape) if tm is not None and hasattr(tm, "shape") else None})
    edges = [{"src": ids[a], "dst": ids[n]} for n in gm.graph.nodes for a in n.all_input_nodes]
    return {"nodes": nodes, "edges": edges}
```
- **Acceptance test** (`tests/test_oracle.py`): for every record in the 5k set, `json.dumps(extract_reference_graph(src), sort_keys=True) == json.dumps(target_graph, sort_keys=True)`. This was verified during spec writing (about 1 minute on CPU). Also test a nested-submodule example, where scope becomes `b1.conv` and name `b1_conv`, and a data-dependent `if`, which must raise `TraceError`.
- **Safety in the app:** user code is `exec`'d, so run the oracle in a **subprocess with a 20 s timeout** (`multiprocessing` with spawn) and catch all exceptions. Report "not traceable: <ErrorType>: <msg>". This is a local demo tool; mention the `exec` caveat in the README.
- Note that `call_method` nodes may have non-tensor outputs (e.g. `size`). In that case `shape` is `None`, and the validator and metrics must tolerate it.

---

## 7. Data preparation (`scripts/prepare_data.py`)

All steps are deterministic with seed 643. Outputs go to `data/processed/`; commit `splits.json` and `dataset_stats.md`.

1. **Load** all 5,000 records into `records.jsonl`, adding `n_nodes`, `n_edges` and `size_bin` (≤10, 11–15, 16–20, >20 nodes).
2. **Structure id.** `struct_id` = `networkx.weisfeiler_lehman_graph_hash(G_unlabelled, iterations=3)` of the DiGraph. Also compute `struct_id_kind` with `node_attr="kind"`. Hashes change across networkx versions, so compute once and **store them in `splits.json`**.
3. **Novel-topology holdout H.**
   - For each tier, list the unlabelled structure classes present in the official train split, sorted by (count desc, struct_id).
   - Pick with a seeded RNG `k_t = max(1, round(0.2 * n_classes_t))` classes, avoiding the single largest class.
   - Expected k is about 1 for L0 (6 classes), 4 for L1, 3 for L2, 2 for L3 and 1 for L4. Log the chosen ids and counts.
   - `novel_topo` eval set = the official-**train** samples whose `struct_id ∈ H`. If more than 120 in a tier, sample whole families (all 4 variants) down to about 120.
   - These samples are removed from fine-tuning data.
4. **Validation split.** From the remaining official-train families, hold out 5% of families per tier (all 4 variants) as `val`. The rest is `train_ft`.
5. **Official test** stays exactly as given: 1,000 samples. Tag each sample `struct_seen = struct_id ∉ H`, since official-test samples with H-structures are also unseen topologies for the fine-tuned model. Report both slices.
6. **Anonymized test** `test_anon`: the official test with `class <Name>(nn.Module)` → `class Net(nn.Module)` (regex, plus any `super(<Name>, self)`). Assert the oracle graph is unchanged on 20 random samples.
7. **SFT files** `sft_train.jsonl` and `sft_val.jsonl` in TRL's **standard (pre-rendered string) prompt-completion** format. This avoids TRL applying a VLM processor template behind our back:
   ```json
   {"prompt": "<chat template rendered with system + user, add_generation_prompt=True, enable_thinking=False>",
    "completion": "<compact CG-JSON><turn-end token>",
    "sample_id": "L2_0003", "tier": "L2"}
   ```
   - The **turn-end token** is whatever the Qwen3.5 template appends after an assistant message (expected `<|im_end|>`).
   - Find it by rendering a full system/user/assistant conversation and taking the suffix after the assistant content. Assert that it is a single special token.
   - Without it, the fine-tuned model may not learn to stop.
   - Build separate SFT files per model family if you fine-tune more than one family.
8. **Eval set files**, each a JSONL with `sample_id, tier, augmentation_type, size_bin, struct_seen, source_code, target_graph, input_shape`:
   - `test.jsonl` (1,000)
   - `test_anon.jsonl` (1,000)
   - `novel_topo.jsonl` (about 400–600)
   - `ood.jsonl` (§7.2)
   - `test_mini.jsonl`: 100 samples, stratified 20 per tier with all augmentations, for quick runs and the P1/P2 ablations
9. **Token audit.** Load the Qwen3.5 tokenizer (`AutoTokenizer`, not the multimodal processor) from local files; this is CPU-only. Compute prompt+completion token lengths for all SFT rows and print max and p99. Set `max_length` = the maximum rounded up to the next multiple of 128, plus 64, and **assert there is zero truncation**. Expect roughly 2k tokens at the maximum, so set the training `max_length` from this audit rather than assuming 2,048.
10. **Stats report** `results/tables/dataset_stats.md`: per-tier counts, node and edge histograms (figure), kind vocabulary, struct counts, split sizes, and the §2.4 findings with numbers.

### 7.2 OOD set (`data/ood/`, `scripts/build_ood.py`), P1 but cheap — do it
Write about **40 small hand-written modules** (roughly 4 per category; the categories below are suggestions, so swap in others that probe topology better), each as a `.py` file with one top-level class plus helper submodule classes. A sidecar `meta.json` gives the `input_shape` and `class_name`. **Input shapes must follow the prompt's convention** (§5.3 rule 7: 16×16 spatial for conv nets, `[1, in_features]` for MLPs, per-input for multi-input models). Otherwise shapes are not predictable from the code, and shape metrics on OOD would be meaningless. Label them with the oracle; any that fail to trace are fixed or dropped. Vary attribute names, forward arg names and channel sizes. Categories:
1. MLP with `nn.LayerNorm`, `nn.GELU` and `nn.Dropout` modules (Linear inputs)
2. VGG-ish: conv → `MaxPool2d` / `AvgPool2d` → `flatten` → Linear head
3. ResNet `BasicBlock` as a **nested submodule class** used twice (dotted scopes)
4. Bottleneck with projection shortcut using `F.relu` (functional calls)
5. Inception-like: 3–4 branches of different depths → `torch.cat`
6. U-Net-lite: encoder/decoder with `F.interpolate` or `nn.Upsample` and a skip `cat`
7. DenseNet-style: each layer consumes the `cat` of all previous outputs
8. Squeeze-and-Excitation: `x.mean((2,3))` → Linear → sigmoid → `x * s.view(...)` (call_method and mul)
9. Two-input model `forward(self, x, y)` with fusion by add or cat (multiple placeholders keep their arg names)
10. `nn.ModuleList` with a static `for` loop (unrolled; names like `layers_0`)

Also write 6 **untraceable** examples (data-dependent `if`/`while`, `assert`, `int(x.shape[1])` coercion) in `data/ood/untraceable/`. They have no reference, so use them only for qualitative app demos and the report; this mirrors prep slide 16. `ood.jsonl` must use the same fields as the other eval sets, with `tier="OOD"` and `category`.

---

## 8. Validity and topological consistency checker (`tgraph/validate.py`)

`validate(cg, source_code=None) -> ValidationReport` with fields `levels_passed: dict[str,bool]`, `errors: list[{code, level, node, msg}]`, `n_errors_by_code`, and `score` (fraction of applicable checks passed). The checks never use the reference graph; that is what makes them usable at inference time for verify-and-retry and in the app.

The checks below are a strong starting set; add or merge checks as you see fit, keeping the levels meaningful.

| Level | Check | Error code |
|---|---|---|
| V0 Parse/schema | JSON parses; schema valid | `PARSE`, `SCHEMA` |
| V1 Structure | names unique | `DUP_NAME` |
| | edge indices in range; no self-loops; no duplicate edges | `BAD_EDGE`, `SELF_LOOP`, `DUP_EDGE` |
| | ≥1 placeholder, exactly 1 output, output is the last node | `NO_INPUT`, `OUTPUT_COUNT` |
| | placeholders have in-degree 0; output has in-degree ≥1 and out-degree 0 | `INPUT_HAS_PRED`, `OUTPUT_DEGREE` |
| | acyclic (`nx.is_directed_acyclic_graph`); the list order is a topological order (all edges i<j) | `CYCLE`, `ORDER` |
| | every node is reachable from some placeholder and reaches the output (no orphans or dead ends) | `ORPHAN`, `DEAD_END` |
| | op/kind consistency (placeholder↔placeholder, output↔output; call_module kind is CamelCase; call_function kind is lowercase) | `OP_KIND` |
| V2 Arity | `add`/`mul`/`sub` ≥1 tensor input, flagged if exactly 1 (warning); `cat`/`stack` ≥2; standard `call_module` layers exactly 1 input; output exactly 1 input | `ARITY` |
| V3 Code consistency (AST, no execution; `tgraph/codeinfo.py`) | each `call_module` name maps to a declared attribute (`self.a.b` → `a_b`, also for `nn.Sequential` and `ModuleList` children, best-effort); otherwise a **hallucinated module** | `UNDECLARED_MODULE` |
| | the declared class matches `kind` | `KIND_MISMATCH` |
| | every attribute **live** in forward (its result flows to the return value) appears as many times as it is called; dead attributes (e.g. `unused_op_*`) must not appear | `MISSING_MODULE`, `DEAD_CODE_INCLUDED` |
| V4 Shapes | Conv2d `shape[1] == out_channels`; Linear `shape[-1] == out_features`; BN, activations, dropout and identity preserve input shape; `add`/`mul` inputs broadcast-compatible and equal to the output; `cat(dim=1)` channels sum; `flatten(1)` is a product; output shape equals its input's shape; placeholder shape follows the §5.3 rule 7 convention unless an explicit input shape is supplied (e.g. from the app's override) | `SHAPE` |

`codeinfo.py`:
- Parse `__init__` assignments `self.<attr> = nn.<Class>(args…)` and record class, positional and keyword args (in/out channels and features). Unknown classes are recorded as opaque.
- Parse `forward`: build a simple def-use chain over straight-line statements (assignments and returns) to find which `self.<attr>` calls are live, by doing a backward slice from the `return`.
- A best-effort implementation is fine. It must be **exact on the dataset**, which is the acceptance test below.

**Acceptance tests (`tests/test_validate.py`):**
- All 5,000 gold graphs (converted to CG) with their source give **zero errors**. If a V3/V4 heuristic produces false positives, fix it. If it cannot be made exact in reasonable time, downgrade that check to a *warning* (excluded from `fully_valid`) and document it; never silently drop gold samples from the test.
- Synthetic corruptions must each be detected by the expected code: delete a skip edge (→ `DEAD_END` or `ARITY` on `add`), add a back edge (`CYCLE`), rename a module to an undeclared name, insert the dead `unused_op` node, change a Conv out-channel shape, and drop the output node.

The validator powers: (a) validity metrics (§9), (b) verify-and-retry feedback messages (§12), and (c) the app's validity panel. Error messages must be human-readable and short, e.g. `node 7 'add' has 1 input; residual add needs 2 (missing skip edge?)`.

---

## 9. Metrics (`tgraph/align.py`, `tgraph/properties.py`, `tgraph/metrics.py`)

All metrics are computed per sample, then macro-averaged per slice. Both graphs are canonical (dataset format). Before comparing, normalize: if a graph has a single placeholder, set its name to `x`.

### 9.1 Node alignment
1. **Name match:** predicted and reference nodes with an identical `name` are matched. fx naming is deterministic, so this is strict and meaningful.
2. **Hungarian fallback** for nodes left unmatched (`scipy.optimize.linear_sum_assignment`). Only nodes of the **same kind** may be paired, with cost = 0.25·|topological level difference|. Pairs with cost ≥ 1 (or different kinds) stay unmatched.
3. The result is a mapping π: pred→ref, partial. Report metrics under **strict** (name-only) and **aligned** (name + Hungarian) alignment. Strict is the headline; aligned shows structure correctness despite naming slips.

### 9.2 Metric definitions (the core set; extend or trim as the results suggest)
- **Validity:**
  - `parse_ok`, `schema_ok`, `struct_valid` (V1), `arity_valid` (V2), `code_consistent` (V3), `shape_consistent` (V4), and `fully_valid` (V1–V4 all pass)
  - `hallucination_rate` = predicted call_module nodes with `UNDECLARED_MODULE`, divided by all predicted call_module nodes
  - `dead_code_inclusion_rate` = samples with `DEAD_CODE_INCLUDED`, over aug_variant_2 and aug_variant_3 samples
- **Node P/R/F1:**
  - strict: matched pairs with equal (name, op, kind) over |V̂| and |V|
  - `kind_multiset_F1`: alignment-free multiset overlap of kinds
- **Edge P/R/F1:**
  - strict: predicted edges (name_u, name_v) vs reference edges, as sets
  - aligned: map predicted edges through π
  - `typed_edge_multiset_F1`: alignment-free, over multisets of (kind_u→kind_v)
- **Exact match:**
  - `exact_strict`: identical node lists (name, op, kind, order-insensitive) **and** identical edge sets
  - `isomorphic`: `nx.is_isomorphic(Ĝ, G, node_match=kind equal)` using DiGraph VF2, which is fast at our sizes. This is order- and name-invariant (permutation invariance, prep slide 5).
  - WL hash equality is computed only as a cheap screen. **It is never used as a distance** (prep slide 10).
- **Shape accuracy:** among strict-matched nodes, the fraction with an identical shape.
- **Graph edit distance:**
  - `aligned_edit_distance` (AED) under π: node cost = unmatched nodes + matched nodes with different kind; edge cost = |E ⊕ π(Ê)|.
  - Normalize: `nAED = AED / (|V|+|E|+|V̂|+|Ê|)`, and **`graph_similarity = 1 − nAED`**. AED is an upper bound on true GED and costs O(n).
  - True GED: on a **stratified 100-sample subset** per config, run `nx.optimize_graph_edit_distance` (node substitution cost 1 when kinds differ, unit insert/delete for nodes and edges), take the best value within a **2 s timeout**, and report it as GED (upper bound if timed out). Report Spearman correlation(AED, GED) to justify AED (figure F8).
- **Topological property preservation** (`properties.py`). Compute these properties on both graphs:
  - `n_nodes`, `n_edges`, `n_sources`, `n_sinks`
  - `n_branch` (out-degree > 1), `n_merge` (in-degree > 1)
  - `depth` (longest path length), `width` (max nodes per topological generation)
  - `n_skip_edges` (edges spanning more than one topological level: residual and long-range dependence)
  - `n_paths_in_out` (number of input→output paths, capped at 10⁴)
  - `is_dag`, `weakly_connected`, `n_cycles` (simple cycles, capped)
  - `hierarchy_groups` (number of name-prefix clusters, §10)

  Report: **Property Preservation Score (PPS)** = mean over properties of 1[equal]; a per-property match rate table; and per-property MAE for counts. This directly encodes "connectivity, adjacency, cycles, hierarchy" from the brief.
- **Connectivity preservation:** for each matched pair (u,v), reachability agreement: the fraction of reference reachable pairs (u ⇝ v) preserved in the prediction, plus precision over predicted reachable pairs. Report `reach_F1`, which captures transitive connectivity even when a single edge is misplaced.

### 9.3 Slices and statistics
- **Slices:**
  - eval set (test, test_anon, novel_topo, ood)
  - tier L0–L4
  - augmentation_type
  - size_bin
  - struct_seen vs unseen
  - OOD category
- **Headline metrics:** fully_valid, edge F1 (strict), exact_strict, isomorphic, graph_similarity, PPS, shape accuracy. Each gets a **bootstrap 95% CI** (1,000 resamples). With a P1 slot, also run a paired comparison between the fine-tuned and base 9B on the same samples (paired bootstrap of the difference).
- **Failure taxonomy** (`scripts/failure_analysis.py`). Classify each non-exact prediction with rule-based labels:
  - `parse_fail`
  - `missing_skip_edge` (ref edge spanning >1 level absent)
  - `extra_edge`
  - `missing_node`
  - `hallucinated_node`
  - `dead_code_included`
  - `wrong_kind`
  - `naming_only` (isomorphic but not strict-exact)
  - `shape_only` (topology exact, shapes wrong)
  - `order_error`
  - `cycle`

  A sample can have several labels. Output counts per config and tier (figure F7), plus 2 example ids per label for the gallery.

**Acceptance tests (`tests/test_metrics.py`):**
- `metrics(g, g)` is perfect for 50 gold graphs.
- Each single corruption lowers edge F1 or exact match as expected.
- A permutation of node order plus renaming keeps `isomorphic=True` while strict metrics drop.
- AED is ≥ true GED on 20 random pairs where GED finishes.

---

## 10. Visualization (`tgraph/visualize.py`)

- `render(graph, path, title=None, fmt="svg"|"png", highlight=None, cluster=True, expand_neurons=False)` uses the `graphviz` Python package with `rankdir=TB`. A suggested style that mirrors the prep deck's taxonomy slide:

  | Node type | Shape | Fill | Label |
  |---|---|---|---|
  | placeholder, output | ellipse | grey | name |
  | call_module | box | light blue | `name\nKind\n[shape]` |
  | call_function, call_method | rounded box | white | `name\n[shape]` |

  Show shapes on a toggle (default on), with font size 10.
- **Hierarchy clusters:** group `call_module` nodes whose names share a prefix before the first `_` or `.` (e.g. `b0_*`, `pa_*`, `branch_a_*`), or share a dotted scope, into `subgraph cluster_<prefix>` with a dashed border. Attach functional nodes (relu, add) to the cluster of their unique predecessor when that predecessor is clustered. Only form a cluster with at least 2 members.
- **Diff overlay** `render_diff(pred, ref, path)`: the union graph under alignment π.

  | Element | Style |
  |---|---|
  | matched edges | solid black |
  | missing edges (in reference only) | **red dashed** |
  | hallucinated edges (in prediction only) | **orange bold** |
  | missing nodes | red outline, dashed |
  | hallucinated nodes | orange fill |
  | wrong kind | red label |
  | wrong shape | shape text in red |

  Include a legend node.
- **Side-by-side** `render_pair(pred, ref, path)`: two graphs in one figure (graphviz `cluster` subgraphs with `rank=same` headers, or combine two PNGs with PIL), titled "Reference (torch.fx)" and "Predicted (<model>)".
- **Neuron expansion** (display-only, for "neurons within layers"): for `Linear(in,out)` with in, out ≤ 8, draw unit nodes `name_u0…` with complete bipartite edges between consecutive layers. Use it for L0 and the MLP OOD category.
- **Fallback** when the `dot` binary is missing: `networkx` layered layout using `multipartite_layout` with `subset = topological generation`, drawn with matplotlib using the same colors. Test both paths.
- `scripts/gallery.py` renders, for the main model:
  - per tier, the best, median and worst samples by graph_similarity on `test`
  - 6 OOD samples
  - 3 untraceable examples (prediction only)

  Each as reference / prediction / diff. It writes `results/figures/gallery/` plus `gallery.md` (and optionally a static `gallery.html`).

---

## 11. Fine-tuning (`scripts/train_sft.py`, env `tg-train`)

**Models.** P0: `Qwen3.5-2B` (fast, scaling point) and `Qwen3.5-9B` (main). Smoke test: `Qwen3.5-0.8B`.

**Method.**
- Supervised fine-tuning with **LoRA in bf16**.
- **Not QLoRA:** 4-bit QLoRA is discouraged for Qwen3.5 because the quantization error is unusually large, and every GPU here has ≥40 GB anyway.
- The loss is token-level cross-entropy on **completion tokens only**. TRL's `SFTTrainer` does this by default for prompt-completion data (`completion_only_loss=None` → completion-only).
- **Text-only:** the vision encoder is frozen and gets no LoRA.

**Loading Qwen3.5 (it is a vision-language model):**
1. Try `AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)` first. If it fails or drops weights, load the official class (`Qwen3_5ForConditionalGeneration` / `AutoModelForImageTextToText`) and train its language model. Log which one worked.
2. Load the tokenizer with `AutoTokenizer` (not the processor) and pass it as `processing_class`. Pass the **loaded model object** to `SFTTrainer`, not a path string, so TRL does not pick a different auto class.
3. Use `attn_implementation="sdpa"` for the full-attention layers. The Gated DeltaNet layers use the `fla` / `causal-conv1d` kernels if installed. Log at startup whether the fast path is active; transformers usually warns when it falls back.
4. **LoRA targets:**
   - Print the set of `nn.Linear` leaf-module names in the language model (exclude anything under the vision tower, usually `visual`) and save it to `DECISIONS.md`.
   - Default targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. This list is known to work for Qwen3.5 SFT in Unsloth's recipe.
   - Restrict it to the language model with a regex `target_modules` string so the vision tower is never adapted, e.g. `r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"`.
   - The linear-attention layers have their own projection names. **Also add those projections** if the printed module list makes them obvious.

Starting hyperparameters (adjust based on the smoke test, val loss and time available):

| Hyperparameter | 2B | 9B |
|---|---|---|
| LoRA r / alpha / dropout | 32 / 64 / 0.05 | 32 / 64 / 0.05 |
| target modules | see above (language model only) | same |
| learning rate, schedule | 2e-4, cosine, warmup_ratio 0.03 | 1e-4, cosine, warmup_ratio 0.03 |
| epochs | 2 | 2 (fewer if the timing rule below says so) |
| per-device batch × grad accum | 8 × 2 (eff. 16) | 2 × 8 (eff. 16) on 40–48 GB; 4 × 4 on 80 GB |
| max_length | from token audit (expected ≤ 2048) | same |
| gradient_checkpointing | True | True |
| packing | False | False |
| eval | `sft_val` loss every 50 steps | same |
| save | every 50 steps, `save_total_limit=2` | same |
| seed | 643 | 643 |
| logging | `report_to="tensorboard"` + CSV of the log history | same |

**Cost and the timing rule.**
- About 3,000 train rows × about 1k tokens ≈ 3M tokens per epoch, so about 190 optimizer steps per epoch at an effective batch of 16.
- Hybrid-attention speed depends heavily on whether the fast kernels are active, so **do not trust a fixed estimate**. Use the GPU smoke test's seconds per step: `ETA = s_per_step × 190 × epochs`.
- Suggested rule: if the 9B ETA for 2 epochs is over about 2.5 h, train 1 epoch. If even that is too long, train on 2 of the 4 augmentation variants per family (`none` + `aug_variant_3`).
- Use judgement based on the actual GPU availability, and log the decision in `DECISIONS.md`. The 2B model is expected to take well under an hour.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
tok = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa")
args = SFTConfig(output_dir=out, max_length=MAXLEN, num_train_epochs=EPOCHS, learning_rate=LR,
                 per_device_train_batch_size=BS, gradient_accumulation_steps=GA,
                 lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
                 gradient_checkpointing=True, logging_steps=10, eval_strategy="steps",
                 eval_steps=50, save_steps=50, save_total_limit=2, report_to="tensorboard", seed=643)
peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=TARGET_REGEX)     # language-model projections only (see above)
trainer = SFTTrainer(model=model, processing_class=tok, args=args,
                     train_dataset=train_ds, eval_dataset=val_ds, peft_config=peft)
trainer.train(resume_from_checkpoint=last_ckpt_or_None)   # resumable after a job time-out
trainer.save_model(f"models/{tag}-lora")                    # adapter + tokenizer
```
- Argument names drift between TRL and transformers versions (`max_seq_length`→`max_length`, `evaluation_strategy`→`eval_strategy`, `torch_dtype`→`dtype` in transformers v5). **Check the installed signatures** and adapt.
- Before the real run, print one tokenized training example with its completion mask. Confirm that:
  - only the CG-JSON and the turn-end token are unmasked
  - the prompt ends exactly where inference prompts end
  - the sequence ends with **exactly one** turn-end/EOS token. Some TRL versions append the tokenizer's EOS to completions automatically; if so, do not also append it yourself.
- **`scripts/merge_lora.py`:**
  - Load the base model in bf16 with the same class used for training, apply the adapter, `merge_and_unload()`, and save to `$MODELS/<tag>-merged`.
  - **Copy every non-weight file from the base model folder** (tokenizer, chat template, `preprocessor_config.json`, generation config, processor files) so vLLM loads it exactly like the base model.
  - If vLLM cannot load the merged folder (e.g. the class or config changed because only the language model was saved), try vLLM's LoRA serving with the base model plus the adapter (`enable_lora=True`, `LoRARequest`). If neither works, use the HF backend for the fine-tuned runs and log it.
  - The deliverable is the **adapter** (small); the merged weights can be regenerated.
- **CPU smoke test (Phase 2, mandatory):** Qwen3.5-0.8B, 16 samples, `max_steps=3`, on a CPU machine (slow pure-PyTorch path, which is fine for 3 steps). It must run end to end, save an adapter, merge it, and generate one sample with the HF backend.
- **Sanity after training:** greedy-decode 20 `val` samples with the HF backend inside the training job and log exact-match. If it is below 50% for 9B, flag it in the log; that indicates a template, mask or turn-end-token bug. The inference job still runs.

## 12. Inference (`scripts/generate.py`, env `tg-infer`)

CLI: `--config <name from experiments.yaml> --eval_sets test,novel_topo,... --backend vllm|hf --limit N`.

**vLLM backend (GPU, default):**
```python
from vllm import LLM, SamplingParams
try:                                   # vLLM >= ~0.10 API
    from vllm.sampling_params import StructuredOutputsParams
    def constrained(schema): return {"structured_outputs": StructuredOutputsParams(json=schema)}
except ImportError:                    # older API
    from vllm.sampling_params import GuidedDecodingParams
    def constrained(schema): return {"guided_decoding": GuidedDecodingParams(json=schema)}

llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=643,
          limit_mm_per_prompt={"image": 0, "video": 0})   # text-only: skip multimodal memory (drop if unsupported)
sp = SamplingParams(temperature=0.0, max_tokens=1536, **(constrained(SCHEMA) if CONSTRAINED else {}))
outs = llm.generate(prerendered_prompts, sp)  # SAME strings as training: chat template, enable_thinking=False
```
- Use `llm.generate` on **pre-rendered prompt strings** (§5.3), not `llm.chat`, so the prompt bytes are identical to training.
- Batch **all** prompts of an eval set in one `generate` call. Save one JSONL row per sample: `sample_id, config, raw_text, n_output_tokens, finish_reason, latency_s(batch-amortized)`, plus the parsed CG and validation report.
- **Thinking-mode ablation (E16):**
  - Render with `enable_thinking=True` and run unconstrained, since a JSON constraint would forbid the reasoning text.
  - Use the sampling parameters recommended on the Qwen3.5 model card for thinking mode, with `max_tokens=8192`, on `test_mini` and `ood` only.
  - Parse the JSON after the closing `</think>` (§5.4 already strips the think block).
  - Log the reasoning tokens used.
- For the 32B-AWQ ceiling, use `quantization="awq"` (or let it auto-detect) with `max_model_len=8192`. For Qwen3.5-27B, use bf16 on an 80 GB A100.
- If vLLM fails to install or run on the cluster, use the **HF backend** below for everything. It is slower, so restrict P1/P2 runs to `test_mini`, and log the switch in `DECISIONS.md`.

**HF transformers backend** (CPU/GPU fallback; used by the app on CPU):
- Batched greedy `generate` with left padding and `max_new_tokens=1536`.
- Optional constrained decoding via xgrammar's HF logits processor. If its API differs, skip it and rely on parse+repair.

**Verify-and-retry (P1, a framework contribution):**
- After the initial generation, run `validate(cg, source_code)`. For samples with errors (any level), append the assistant output and a user turn:
  `"Your graph has these problems:\n- <error msg 1>\n- ...\nReturn the corrected full JSON graph."`
- Regenerate, constrained, for up to **R=2** rounds. Keep the last valid output or the best-scoring one.
- Render the multi-turn conversation with the same chat template and `enable_thinking=False`. The fine-tuned model was trained only on single-turn data, so the retry may help it less than the base model. That itself is a finding worth reporting.
- Log `rounds_used`. This uses **only** the validator, never the reference, so it is legitimate at test time.

**Self-consistency (P2):** N=5 samples at temperature 0.7. Choose the candidate with the highest validator score, breaking ties by edge-majority vote across candidates.

**`configs/experiments.yaml`** defines every run (see §15). Example entry:
```yaml
- name: ft9b_c
  model_dir: ${MODELS}/qwen35-9b-topograph-merged
  prompt_mode: zero_shot        # zero_shot | few_shot
  constrained: true
  retry_rounds: 0
  eval_sets: [test, test_anon, novel_topo, ood]
```

---

## 13. Evaluation pipeline (`scripts/run_eval.py`, `aggregate.py`, `plots.py`)

- `run_eval.py --pred results/raw/<config>/<set>.jsonl` joins predictions with references and computes §8 and §9 per sample using a `multiprocessing.Pool`. It writes `<set>.metrics.jsonl`. It is CPU-only (run it on any CPU machine or as `jobs/cpu_eval.sh`).
- `aggregate.py` writes:
  - `results/tables/main.csv` and `main.md`: configs × headline metrics on `test`, with CIs
  - `by_tier.csv`, `by_aug.csv`, `by_size.csv`, `seen_vs_unseen.csv`, `ood_by_category.csv`, `properties.csv` (per-property match rates), `validity.csv`, `failures.csv`, `retry.csv`
- `plots.py` writes these figures (PNG + SVG, matplotlib, consistent colors per config, readable fonts, every axis labelled):

  | Figure | Content |
  |---|---|
  | F1 | Headline metrics by config (grouped bars with CI whiskers) |
  | F2 | Edge F1 and exact-match vs tier L0→L4, one line per config (**the "complexity" result**) |
  | F3 | graph_similarity vs size_bin |
  | F4 | Seen vs unseen topology (official test seen / test unseen-H / novel_topo) for base vs FT (**the memorization vs generalization result**) |
  | F5 | Robustness by augmentation type and anonymized vs original class name |
  | F6 | Validity rates, constrained vs unconstrained, and before vs after retry |
  | F7 | Failure taxonomy stacked bars per config |
  | F8 | AED vs true GED scatter (metric validation) |
  | F9 | Training and validation loss curves for 2B and 9B |
  | F10 | Per-property preservation heatmap (configs × properties) |

- `gallery.py` (§10) produces the qualitative comparisons.

---

## 14. Interface: Gradio app (`app/app.py`, env `tg-infer`)

**Suggested layout (`gr.Blocks`, title "TopoGraph — LLMs drawing neural-network topology"); the required parts are code input, model choice, predicted graph, reference graph (when traceable), comparison and validity:**
- **Inputs:**
  - code editor (`gr.Code(language="python")`) and `.py` file upload; the upload fills the editor
  - an "Examples" dropdown with about 10 picks: 2 per tier from `test`, 3 OOD, and 2 untraceable
  - model dropdown: `FT-9B`, `FT-2B`, `Base-9B (few-shot)`, `Oracle only`; only the models whose files exist
  - checkbox "Constrained decoding" (default on), checkbox "Verify & retry" (default on)
  - optional "Input shape override" textbox, e.g. `1,3,32,32`, and optional "Top-level class name". The override goes to the oracle and the validator, and is also appended to the user prompt as `Input shape: ...`. The fine-tuned model never saw that line in training, so label predicted shapes as "best effort" when an override is used.
  - "Generate graph" button
- **Outputs (tabs):**
  1. **Predicted graph** (SVG/PNG, hierarchy clusters, toggle neuron expansion)
  2. **Reference (torch.fx)**, or a clear message "Code not traceable by torch.fx (<error>): no reference; showing LLM prediction only". This is the scientifically interesting case.
  3. **Diff overlay** with legend (only when a reference exists)
  4. **Metrics** table: node/edge F1, exact, isomorphic, graph similarity, PPS, shape accuracy, plus the property comparison table
  5. **Validity report**: pass/fail per level and the error list with messages; retry rounds used
  6. **JSON**: predicted and reference graphs with a download button
- **Batch tab:** upload a JSONL or JSON list of `{source_code[, target_graph]}` and get a downloadable CSV of metrics plus a summary.
- **About tab:** short method description and a link to the README.

**Backends and modes:**
- **GPU mode** (on a GPU machine, the real demo): vLLM with the merged FT model. Load once at startup, with ~0.5 gpu_memory_utilization if two models are loaded.
- **CPU mode** (any CPU machine): HF backend with `FT-2B` (slow but works, roughly 15–90 s per graph on CPU because the linear-attention layers use the pure-PyTorch path), or a **"Precomputed" mode** that looks up stored predictions for dataset or OOD examples by hash of the source. The demo always works without a GPU.
- The oracle always runs in a subprocess with a timeout (§6).

**Launching:** `demo.launch(server_name="0.0.0.0", server_port=7860)`; do not rely on `share=True`, since the GPU machine may have no internet. Provide `scripts/run_app.sh --mode gpu|cpu|precomputed`. How to reach the app on Prajna (interactive allocation, port forwarding) comes from the user's Prajna doc; copy those steps into the README.

**Acceptance:**
- In CPU precomputed mode, every example renders all tabs without errors.
- An untraceable example shows the prediction plus the "no reference" message.
- Bad input (syntax error, no nn.Module) shows a friendly error and does not crash.
- Write a short `tests/test_app_smoke.py` that calls the core handler function directly; do not test the UI.

---

## 15. Experiment matrix (priorities and GPU cost)

All runs use greedy decoding unless noted. "Sets" refers to the eval sets in §7.

- **"base" here means the released Qwen3.5 checkpoint (`Qwen/Qwen3.5-9B`, the instruct/hybrid model the user already runs) without our fine-tuning.** It does **not** mean the separate `-Base` pretrained checkpoints.
- P0 rows are the core comparison. P1/P2 rows are suggestions: run, replace or add experiments based on what the early results reveal. For example, if retry clearly helps, give it more sets; if constrained vs unconstrained shows no difference, spend the time elsewhere.

| ID | Pri | Config | Model | Prompt | Decoding | Sets | Est. GPU time |
|---|---|---|---|---|---|---|---|
| E1 | P0 | `base9b_zs_c` | Qwen3.5-9B base | zero-shot | constrained | test, novel_topo, ood | ~10 min |
| E2 | P0 | `base9b_fs_c` | Qwen3.5-9B base | 3-shot | constrained | test, test_anon, novel_topo, ood | ~15 min |
| E3 | P0 | `base2b_fs_c` | Qwen3.5-2B base | 3-shot | constrained | test, novel_topo, ood | ~5 min |
| E4 | P0 | `ft2b_c` | Qwen3.5-2B FT | zero-shot | constrained | test, test_anon, novel_topo, ood | ~5 min |
| E5 | P0 | `ft9b_c` | Qwen3.5-9B FT | zero-shot | constrained | test, test_anon, novel_topo, ood | ~10 min |
| E6 | P0 | `ft9b_u` | Qwen3.5-9B FT | zero-shot | **unconstrained** | test, novel_topo, ood | ~10 min |
| E7 | P0 | `base9b_fs_u` | Qwen3.5-9B base | 3-shot | **unconstrained** | test | ~10 min |
| E8 | P1 | `ft9b_c_retry` | Qwen3.5-9B FT | zero-shot | constrained + verify&retry (R=2) | test, novel_topo, ood | ~15 min |
| E9 | P1 | `base9b_fs_c_retry` | Qwen3.5-9B base | 3-shot | constrained + retry | test_mini, ood | ~5 min |
| E10 | P1 | `ceil_fs_c` | Qwen3.5-27B (80 GB) or Qwen2.5-Coder-32B-AWQ | 3-shot | constrained | test, novel_topo, ood | ~25 min |
| E11 | P1 | `ceil_zs_c` | same ceiling model | zero-shot | constrained | test_mini, ood | ~5 min |
| E16 | P1 | `base9b_think` | Qwen3.5-9B base, **thinking mode** | zero-shot | unconstrained, reason then JSON | test_mini, ood | ~15 min |
| E17 | P1 | `coder7b_fs_c` | Qwen2.5-Coder-7B base | 3-shot | constrained | test, novel_topo, ood | ~10 min |
| E12 | P2 | `ft2b_cot` | Qwen3.5-2B FT on CoT-prefixed targets | zero-shot | unconstrained | test, ood | +10–20 min train |
| E13 | P2 | `base9b_fs_edgelist` | Qwen3.5-9B base | 3-shot, edge-list representation | unconstrained | test_mini | ~5 min |
| E14 | P2 | `ft9b_sc5` | Qwen3.5-9B FT | zero-shot | self-consistency N=5 | test_mini, ood | ~10 min |

Packaging of runs into jobs:
- `infer_baselines.sh`: E1, E2, E3, E7 (plus E9 and E16 if P1).
- `train.sh`: 2B then 9B, then merge both.
- `infer_finetuned.sh`: E4, E5, E6, E8.
- `infer_ceiling.sh`: E10, E11, E17.

Total P0+P1 GPU time is about **4–5 h on an A100**, or about 6 h on an A40 or L40. The largest uncertainty is 9B training speed (§11 timing rule).

Reference line for every table: the **oracle (torch.fx)** scores 100% on all traceable sets by construction. Say this explicitly; it is the deterministic-analyser baseline from prep slide 8.

---

## 16. Indicative timeline and run order (T0 = when Claude Code starts; adapt to reality)

| Time | Phase | Work | Gate before moving on |
|---|---|---|---|
| T0 – T+0:45 | **0. Setup** | set up envs (§3.2) and get the models (§3.3) following the user's Prajna doc, with long installs and downloads in the background; confirm the dataset is in `data/raw/` (§3.4); repo skeleton; git init | envs building; downloads running |
| T+0:45 – T+3:00 | **1. CPU core** | `oracle.py` + test (5k exact); `schema.py` round-trip; `prompts.py`; `prepare_data.py` (splits, SFT files, token audit, stats); `codeinfo.py` + `validate.py` + tests; `align/properties/metrics.py` + tests; `visualize.py` (both backends) | all pytest green; `dataset_stats.md` written; token audit passes |
| T+3:00 – T+3:30 | **2. Smoke** | CPU smoke: Qwen3.5-0.8B train 3 steps → merge → HF generate → parse → validate → metrics → render. Then run `gpu_smoke.sh` (includes the 9B timing run) | end-to-end works on CPU; GPU smoke OK; 9B epochs decided (§11) |
| T+3:30 | **3. Launch GPU** | start `infer_baselines.sh` and `train.sh`; `infer_finetuned.sh` runs after `train.sh` succeeds; `infer_ceiling.sh` (P1) | jobs running or queued; ids in `TODO.md` |
| T+3:30 – T+7:00 | **4. While GPU runs** | OOD set (§7.2); `run_eval/aggregate/plots/gallery/failure_analysis`, tested on baseline outputs as they land; Gradio app (precomputed + HF CPU mode first); README skeleton | app works in CPU mode; eval scripts run on baseline outputs |
| T+7:00 – T+9:00 | **5. Evaluate** | when FT inference finishes: `cpu_eval.sh` → all tables and figures; gallery; failure analysis; sanity-check that numbers are plausible (FT ≥ base on seen-topology test; investigate if not) | `results/tables/main.md` filled |
| T+9:00 – T+11:00 | **6. App + package** | GPU demo session with the FT-9B model, screenshot the app for the README; precompute predictions for app examples; README, MODEL_CARD, requirements; package adapters; final `pytest`; `REPORT_NOTES.md` | §17 checklist complete |
| buffer | | GPU delays: if GPU access slips a lot, cut P1/P2 experiments and keep P0 | |

If only **one** GPU job can run at a time, use a single combined script in this order: GPU smoke → E1/E2/E3/E7 → train 2B → E4 → train 9B → E5/E6 → P1 items. That way every completed stage is already useful.

---

## 17. Deliverables and packaging

- **`README.md`**:
  - overview and figure of the pipeline (the four phases from prep slide 12: dataset → fine-tuning → inference with constrained decoding and validator → evaluation and visualization)
  - repo map
  - installation (both envs, CUDA note)
  - data setup
  - how to reproduce (exact commands for prepare → train → generate → eval → plots)
  - how to run the app (GPU, CPU and precomputed modes)
  - main results table (copied from `results/tables/main.md`) with 3–4 key figures
  - limitations (§2.4, oracle scope = statically traceable code, `exec` caveat)
  - team and acknowledgements
- **`MODEL_CARD.md`** for each adapter: base model, LoRA config, training data and splits, hyperparameters, training time and GPU, eval results, intended use, limitations.
- **Models:**
  - `models/qwen3.5-9b-topograph-lora/` and `models/qwen3.5-2b-topograph-lora/`: adapter weights, tokenizer, `adapter_config.json`, and the training-args JSON
  - `scripts/merge_lora.py` to rebuild merged weights
  - **Ask the user** whether the submission needs merged full weights (about 19 GB for 9B) or whether adapters plus the merge script are acceptable, and whether to upload to the Hugging Face Hub.
- **Dependencies:** `requirements-train.txt` and `requirements-infer.txt` (direct deps with versions) plus the `pip freeze` lock files; Python version; CUDA/driver used.
- **Results:** `results/tables/`, `results/figures/`, `results/figures/gallery/`.
- **Submission zip** (`scripts/package.sh`): code + configs + `data/processed/splits.json` + `data/ood/` + results + adapters. Exclude raw data copies unless asked, and exclude merged weights.
- **`REPORT_NOTES.md`**: bullet-point findings with the exact numbers and figure references, ready for the user's report and slides:
  - complexity trend
  - FT vs base vs the ceiling model; Qwen3.5 vs Qwen2.5-Coder; thinking vs non-thinking
  - seen vs unseen topology (memorization)
  - robustness to renaming, dead code and class names
  - constrained vs unconstrained
  - retry gains
  - common failure modes
  - untraceable-code demos

Final checklist: `pytest` green; every P0 run present in `main.md`; the app launches in CPU precomputed mode from a fresh clone following the README; adapters load and generate.

---

## 18. Risks and fallbacks

| Risk | Fallback |
|---|---|
| GPU access is delayed | Launch early (T+3:30); use whichever ≥40 GB GPU is free, since all fit; combined single-job script (§16); cut to P0 |
| vLLM incompatible with the GPU driver/CUDA | Pin an older vLLM or a cu12x torch (§3.2); otherwise use the HF backend everywhere and run P1/P2 on `test_mini` only |
| No `dot` binary | `conda install -c conda-forge graphviz`; otherwise the matplotlib fallback renderer |
| TRL/PEFT API differences | Inspect the installed signatures; adapt the argument names; log it in `DECISIONS.md` |
| Training job hits its time limit | `save_steps=50` + `resume_from_checkpoint`; resubmit the same script |
| 9B FT exact-match suspiciously low | Check the chat template, completion masking and the identical prompt at train and inference; compare HF vs vLLM outputs on 5 samples |
| Constrained decoding very slow or schema unsupported | Simplify the schema (drop `pattern`/`minimum`); otherwise run unconstrained + repair and note it |
| Qwen3.5 training is very slow (no fast linear-attention kernels) | Timing rule in §11: 1 epoch, then half the augmentation variants; the 2B model still gives a fine-tuned result. If 9B cannot finish, report 9B zero/few-shot + 2B FT and say so |
| Qwen3.5 loads as a vision-language model and TRL or PEFT misbehave | Load the model object yourself (§11), restrict LoRA to the language model, pass `processing_class=AutoTokenizer`; check that the vision tower has zero trainable parameters |
| Merged Qwen3.5 model will not load in vLLM | vLLM LoRA serving with the base model plus the adapter; otherwise the HF backend for the FT runs (§11) |
| The model emits `<think>` text or ignores the JSON format | Confirm `enable_thinking=False` in the rendered prompt; the parser strips think blocks; constrained decoding forces JSON |
| GED too slow | AED for all; true GED only on the 100-sample subset with a 2 s timeout |
| Validator false positives on gold | Fix the heuristic; if that is not feasible in time, downgrade that check to a documented warning (§8) |
| CPU-heavy eval is slow on a shared machine | Run eval as `jobs/cpu_eval.sh` with multiprocessing, or on the laptop |

---

## 19. Questions that are the user's to answer

1. **Phase 0:** anything the user's Prajna doc does not cover; please place the dataset zip in `data/raw/` if it is missing.
2. **Before packaging:** submission format: adapters only or merged weights; upload to the HF Hub or a zip; whether the final report or slides are also needed (then generate figures at slide-friendly 16:9 sizes).
3. Any time a fixed item in §0.1 seems to need changing.

---

## 20. Notes for the write-up (what the evidence will support)

- **Complexity:** expect accuracy to drop from L0/L1 to L2–L4, mainly from long-range skip edges (residual adds) and multi-input merges (`cat`). Use F2 and the failure taxonomy (`missing_skip_edge`) to show this. It matches the graph-reasoning literature in prep slide 7.
- **Fine-tuning vs memorization:** the official test only has *seen* topologies (§2.4). Compare FT gains on seen vs novel_topo vs OOD (F4). A large drop on novel topologies means template memorization. A small drop means the model learned topological reasoning. Report this honestly either way.
- **Constrained decoding** guarantees well-formed JSON, not correct content (prep slide 9). Report validity V1–V4 and correctness separately. Retry gains measure how much the *validator* contributes.
- **Reasoning mode:** E16 tests whether "thinking before answering" helps topology extraction, i.e. a test of chain-of-thought at the model level (compare to E1 on the same samples). Report the accuracy gain against the extra tokens used.
- **Robustness:** forward-arg renaming, dead code and anonymized class names test whether the model reads dataflow or surface cues. Dead-code inclusion rate is a direct "hallucination" metric.
- **torch.fx** is exact where applicable. The LLM's value is on untraceable or partial code. Show the untraceable gallery as qualitative evidence, not as a quantitative claim, since it has no reference.

### References (from the prep presentation)
1. Reed et al., *torch.fx: Practical Program Capture and Transformation for Deep Learning in Python*, MLSys 2022. arXiv:2112.08429
2. Wang et al., *Can Language Models Solve Graph Problems in Natural Language?* (NLGraph), NeurIPS 2023. arXiv:2305.10037
3. Tang et al., *GraphArena*, ICLR 2025. arXiv:2407.00379
4. Demirci, Kerur, Singh, *Are LLMs Truly Graph-Savvy?*, ACL SRW 2025
5. Zhang & Kishi, *Finetuning Lightweight LLMs for Control Flow Graph Generation*, arXiv 2026 (2607.04582)
6. Tooling docs: vLLM structured outputs (`StructuredOutputsParams(json=…)` via `SamplingParams(structured_outputs=…)`), TRL `SFTTrainer` (prompt-completion datasets, completion-only loss by default, `max_length`), PEFT LoRA, NetworkX GED/VF2.
