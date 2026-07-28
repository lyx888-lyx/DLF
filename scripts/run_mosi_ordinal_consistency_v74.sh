#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR=./result/complementarity_v71/mosi/seed_1111
CACHE_PATH="$RUN_DIR/complementarity_v71_teacher_cache.pth"

if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[V7.4] Missing teacher cache; rebuilding it from the V7.1 summary..."
  python3 scripts/rebuild_v71_teacher_cache.py \
    --dataset mosi \
    --seed 1111 \
    --gpu 0 \
    --run-dir "$RUN_DIR" \
    --batch-size 32 \
    --num-workers 1
fi

python3 train_ordinal_complementarity_v7_4.py \
  --dataset mosi \
  --seed 1111 \
  --gpu 0 \
  --source-run-dir "$RUN_DIR" \
  --save-root ./result/ordinal_consistency_v74 \
  --batch-size 32 \
  --num-workers 1 \
  --ordinal-hidden-dim 128 \
  --dropout 0.18 \
  --max-ordinal-blend 0.30 \
  --initial-ordinal-blend 0.02 \
  --max-epochs 12 \
  --early-stop 4 \
  --learning-rate 3e-4 \
  --weight-decay 1e-3 \
  --valid-mae-tolerance 0.001 \
  --fold-mae-tolerance 0.010 \
  --min-accuracy-gain 0.002 \
  --folds 3 \
  --boundary-temperature 4.0 \
  --bootstrap-samples 2000
