#!/bin/bash
# Submission zip: code, configs, splits, OOD sets, synthetic data, results, explainer, LoRA adapters.
# Excludes the raw dataset, merged weights and raw predictions (results/raw, 226 MB).
#   scripts/package.sh [out.zip]          (default ../IE643_MachineUnlearning_TopoGraph.zip)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(realpath -m "${1:-../IE643_MachineUnlearning_TopoGraph.zip}")
FILES=(README.md MODEL_CARD.md DECISIONS.md PIPELINE.md TODO.md requirements-*.txt
       configs tgraph scripts jobs app tests docs
       data/ood data/ood2 data/processed/splits.json data/processed/dataset_stats.md data/processed/token_audit.json
       data/processed/ood.jsonl data/processed/ood2.jsonl data/processed/untraceable.jsonl
       data/processed/synthetic.jsonl
       results/tables results/figures runs/*/log_history.csv
       models/qwen35-9b-topograph-aug-lora models/qwen35-9b-topograph-lora
       models/qwen35-2b-topograph-aug-lora models/qwen35-2b-topograph-lora)
[ -f REPORT_NOTES.md ] && FILES+=(REPORT_NOTES.md)
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
trap 'rm -f setup_envs.sh tgenv.sh' EXIT
cp ../setup_envs.sh ../tgenv.sh .          # env scripts live one level up; ship them inside the zip
FILES+=(setup_envs.sh tgenv.sh)
rm -f "$OUT"
zip -qr "$OUT" "${FILES[@]}" -x '*__pycache__*' '*.pyc' '*.pytest_cache*' 'results/figures/smoke/*' \
    'docs/explainer/*.aux' 'docs/explainer/*.log' 'models/*/training_args.bin'
ls -lh "$OUT"
unzip -l "$OUT" | tail -1
