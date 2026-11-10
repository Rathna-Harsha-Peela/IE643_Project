# TODO (phases from PIPELINE.md §16)

## Phase 0 — Setup
- [x] Repo skeleton, git init, dataset unzipped (5,000 files)
- [x] Models: Qwen3.5-0.8B, Qwen3.5-2B downloaded; 9B linked (read-only); Coder-7B downloaded
- [x] Envs `tg-train` (torch 2.11+cu128, fla 0.5.2) and `tg-infer` (torch 2.11+cu129, vLLM 0.26.0+cu129) built

## Phase 1 — CPU core
- [x] oracle + 5k exact test
- [x] schema round-trip, parse
- [x] prepare_data: splits, eval sets, SFT files, token audit, stats
- [x] codeinfo + validate + tests (0 errors on 5k gold; corruptions caught)
- [x] align / properties / metrics + tests
- [x] visualize (dot + mpl) + tests

## Phase 2 — Smoke
- [~] CPU smoke: data/parse/validate/metrics/render path done; CPU training abandoned (Qwen3.5 linear attention too slow on CPU)
- [x] gpu_smoke.sh (vLLM + HF 2-sample gen, 9B 20-step timing)

## Phase 3+ — GPU runs (need explicit "submit" from the user)
- [x] infer_baselines (E1 E2 E3 E7)
- [x] train 2B, 9B + merge; + synthetic-data arm (2B, 9B)
- [x] infer_finetuned (E4 E5 E6 E8)
- [x] P1: ceiling (27B-FP8) / coder7b / thinking / think-then-constrain / retry / self-consistency; OOD-2
- [ ] not run: E12 (CoT targets), E13 (edge-list representation)

## Phase 4–6 — Eval, app, packaging
- [x] OOD set (~40 modules + 6 untraceable)
- [x] run_eval / aggregate / plots / gallery / failure_analysis / GED check
- [x] Gradio app (precomputed + CPU + GPU modes) + smoke test; live GPU demo tested
- [x] README, MODEL_CARD, package.sh, explainer (docs/explainer)
- [ ] REPORT_NOTES (when asked)
