#!/bin/bash
# Metrics for every prediction file, then tables, figures, failure analysis, galleries, app cache. CPU only.
source jobs/common.sh train
export CUDA_VISIBLE_DEVICES=""
P=${SLURM_CPUS_PER_TASK:-16}
python scripts/run_eval.py --all --force --procs $P
python scripts/ged_subset.py
python scripts/aggregate.py > logs/aggregate_stdout.txt
python scripts/failure_analysis.py > /dev/null
python scripts/plots.py
for c in ft9b_c ft9b_aug_c; do python scripts/gallery.py --config $c; mv results/figures/gallery/gallery.md results/figures/gallery/gallery_$c.md; done
python scripts/build_app_cache.py
echo "EVAL_DONE"
