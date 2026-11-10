# DECISIONS

Deviations from / choices within PIPELINE.md, with reasons. Newest at the bottom.

1. **Envs and isolation (2026-10-24).** Both envs live in `~/IE643_Project/envs/` (`conda create -p`), and all
   caches (conda pkgs, pip, HF_HOME, TRITON_CACHE_DIR, TMPDIR) are redirected into `~/IE643_Project/` by
   `tgenv.sh`. Reason: the user's separate KV-compression project relies on its own envs, `~/hf` and `~/.triton`
   staying byte-identical. Qwen3.5-9B weights are reused read-only through a symlink `models/Qwen3.5-9B`.
2. **vLLM version.** GPU driver 570 = CUDA 12.8. vLLM >= 0.27 pins torch 2.13 (no cu128 build). vLLM 0.26.0 is the
   newest release pinned to torch 2.11; it has no cu128 wheel, so `tg-infer` uses torch 2.11.0+cu129 + vLLM
   0.26.0+cu129 (CUDA minor-version compatibility; verified in the GPU smoke test). Spec asked for vLLM >= 0.17: OK.
3. **Training env** uses torch 2.11.0+cu128 + flash-linear-attention 0.5.2, the combination already known to run
   Qwen3.5 on this cluster. `causal-conv1d` is not installed (needs a CUDA compile); transformers' fallback is used
   for the short conv in Gated DeltaNet layers.
4. **Model downloads** are plain folders fetched with curl into `~/IE643_Project/models/<name>` (loaded by path,
   offline on compute nodes) instead of the HF cache layout.
5. **Structure count.** With networkx 3.6.1 the dataset has **54** distinct unlabelled topologies (L0 6, L1 21, L2 14,
   L3 12, L4 4 — per-tier counts match the spec exactly; the spec's "~68" total double-counted structures that occur
   in more than one tier). Hashes are stored in `data/processed/splits.json`.
6. **Novel-topology holdout** (seed 643): k = 1/4/3/2/1 classes for L0..L4; `novel_topo` = 564 official-train
   samples (whole families, capped ~120 per tier); all train samples with a held-out structure are removed from
   fine-tuning. 200 official-test samples have a held-out structure (`struct_seen=False`).
   Split sizes: train_ft 3,068, val 160, test 1,000, test_anon 1,000, test_mini 100.
7. **Token audit:** max prompt+completion 2,127 tokens (p99 2,057) → `max_length` = 2,240, zero truncation.
   Few-shot prompts up to 3,077 tokens. Turn-end token `<|im_end|>` (id 248046). Qwen3.5 0.8B/2B/9B chat
   templates are byte-identical, so one set of SFT files serves all three.
8. **Canonical rendered prompt** ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n` (the template pre-fills an
   empty think block for `enable_thinking=False`); see `data/processed/token_audit.json` for the full example.
9. **Oracle sandbox.** `safe_reference_graph` uses one long-lived *spawned* worker process instead of one process per
   call: a fresh spawn re-imports torch (> 20 s on Lustre), and `fork` deadlocks in torch's CPU thread pool after the
   parent has used it. The worker is restarted if a call exceeds the 20 s timeout.
10. **OOD set:** 40 modules / 10 categories, all traced; the validator gives zero errors on all 40 gold graphs.
    `se_3` got a conv stem: without it rule 7 is ambiguous (the first parametric layer reached is a Linear after
    `x.mean`). 6 untraceable modules (TraceError / TypeError in torch.fx).
11. **Validator:** `add/mul/sub` with one input is an error unless the forward contains a tensor–scalar binary op
    (then it is a warning). Zero errors on all 5,000 gold graphs; all six synthetic corruptions detected.
11b. **vLLM prefix caching off** by default: for Qwen3.5 vLLM 0.26 marks prefix caching on its Mamba-style (linear-attention) layers as experimental ("align" mode), so headline runs do not depend on it.
12. **Failure taxonomy** gains `missing_edge` (a reference edge spanning one level is absent): the spec's list only had `missing_skip_edge`, so a dropped chain edge got no label at all (found in the pipeline dry run).
13. **Train/inference prompt identity across envs:** the infer env (transformers 5.20.2) renders byte-identical prompts to the SFT files built in the train env (5.19.0): 200/200 for both 9B and 2B tokenizers.
14. **Prompt v2 (2026-10-24 23:40, approved by the user).** The first real zero-shot run (9B, test) had 957/1000 outputs
    failing only `OP_KIND` because the spec's prompt never says the input node's kind is "placeholder" (models
    wrote "input"), which also made zero-shot exact match impossible; 12% were cut off at 1,536 tokens by
    pretty-printed JSON. Rule 1 now spells out `{"op":"placeholder","kind":"placeholder"}` (and that multi-input
    placeholders use argument names), the final line asks for compact single-line JSON, and `max_tokens` is 3,072.
    One prompt is still used for every config and for training. SFT files regenerated (splits.json byte-identical,
    max 2,157 tokens ≤ max_length 2,240). The 2B training (30% done), both baseline jobs and the dependent job were
    cancelled and resubmitted; the 9B training was held before it started. Prompt-v1 outputs (base9b_zs_c test,
    base9b_fs_c_retry test_mini/ood) are kept, unreported, in `cache/superseded_prompt_v1/`.
15. **Thinking mode (E16).** With the spec's 8,192-token budget, 97/100 test_mini answers were still reasoning when
    cut off. The parser had taken JSON fragments from inside the unfinished reasoning; for thinking configs an
    answer now requires `</think>` (otherwise "no answer"), applied to stored outputs with `scripts/reparse.py`.
    Added **E16b `base9b_think32k`** (30,000-token budget) so the report can separate "thinking does not help" from
    "thinking did not get to answer". Both are reported.
16. **Oracle worker warm-up:** the sandboxed oracle worker now runs one tiny trace before reporting ready, so lazy first-use imports (slow on a loaded Lustre login node) do not count against the 20 s per-call limit (an app test had failed on that). `run_eval --all` skips `untraceable.jsonl` (no reference exists).
17. **Night improvement plan (user, 25 Oct ~02:00: "improve results overnight, conclusion by 10 am").**
    (a) **E16c `base9b_think32k_c`**: thinking, then JSON constrained to the schema *after* `</think>` (vLLM
        `reasoning_parser="qwen3"`; grammar not enforced inside the reasoning) — targets E16b's missing shapes.
    (b) **E14 `ft9b_sc5`**: self-consistency, 5 samples at T=0.7, validator picks.
    (c) **"aug" arm** (`ft9b_aug_c`, `ft2b_aug_c`): LoRA on official train_ft + ~3,000 synthetic modules from
        `scripts/gen_synthetic.py` (random compositions of general PyTorch blocks, random names), labelled by the
        oracle. Integrity: synthetic graphs whose unlabelled structure is a novel-topology/OOD structure, or whose
        kind-labelled structure equals an OOD graph's, are discarded; nothing from any eval set is used. Caveat for
        the write-up: the same person (Claude) wrote the OOD set and the generator, so the operator vocabulary
        overlaps (e.g. pooling, LayerNorm, upsampling); OOD gains of the aug arm measure transfer to unseen
        hand-written code with a familiar operator vocabulary. The P0 fine-tunes (ft2b_c, ft9b_c) are unchanged and
        remain the main comparison; aug is reported as a separate arm.
18. **Ceiling model (E10/E11) = Qwen3.5-27B-FP8** (official FP8 checkpoint, 29 GB): same family as the main model and fits one L40S; bf16 27B (55.6 GB) does not, and Coder-32B-AWQ is an older family.
19. **OOD-2 set** (`data/ood2/`, 21 modules after the oracle check, 7 categories), written at 03:50 *after* the
    synthetic generator was frozen, using constructs absent from both the dataset and the generator: functional
    pooling, view/reshape/permute, matmul+softmax attention, ConvTranspose2d decoders, GroupNorm/PReLU/ELU/Hardswish,
    torch.stack + reduce, torch.add, scalar arithmetic, F.gelu/silu/log_softmax/normalize/pad/leaky_relu. The 3
    `chunk`/`split` modules were dropped: tuple outputs have no tensor shape (a limitation of the output schema).
    Validator: zero errors on all 21 gold graphs. Purpose: an honest generalisation test for the aug arm, since the
    same author wrote OOD-1 and the generator.
