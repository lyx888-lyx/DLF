#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
RUN_DIR="${RUN_DIR:-./result/complementarity_v71/mosi/seed_${SEED}}"
SAVE_ROOT="${SAVE_ROOT:-./result/positive_residual_specialist_v92}"
CACHE_PATH="$RUN_DIR/complementarity_v71_teacher_cache.pth"

if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[V9.2] Missing teacher cache; rebuilding it from the frozen V7.1 run..."
  python3 scripts/rebuild_v71_teacher_cache.py \
    --dataset mosi \
    --seed "$SEED" \
    --gpu "$GPU" \
    --run-dir "$RUN_DIR" \
    --batch-size 32 \
    --num-workers 1
fi

# First diagnostic run:
# - frozen V7.1 backbone and Student;
# - bounded non-negative correction only;
# - direct residual target from frozen V7.1 hybrid under-prediction;
# - ordinary-positive Train boost, but no ordinary-positive Valid selection term;
# - compact attribution after the specialist checkpoint is frozen.
python3 train_positive_residual_specialist_v9_2.py \
  --dataset mosi \
  --seed "$SEED" \
  --gpu "$GPU" \
  --source-run-dir "$RUN_DIR" \
  --save-root "$SAVE_ROOT" \
  --batch-size 32 \
  --num-workers 1 \
  --legacy-hidden-dim 128 \
  --hidden-dim 160 \
  --dropout 0.18 \
  --max-correction 0.50 \
  --initial-gate-probability 0.20 \
  --initial-magnitude-fraction 0.30 \
  --strong-threshold 1.0 \
  --neutral-radius 1e-6 \
  --region-weight-mode inverse_sqrt \
  --region-weight-min 0.50 \
  --region-weight-max 2.00 \
  --ordinary-positive-boost 1.50 \
  --huber-delta 0.30 \
  --residual-margin 0.05 \
  --max-epochs 30 \
  --early-stop 8 \
  --learning-rate 3e-4 \
  --weight-decay 1e-3 \
  --valid-mae-tolerance 0.001 \
  --fold-mae-tolerance 0.010 \
  --worst-region-tolerance 0.10 \
  --nonpositive-mae-tolerance 0.010 \
  --robust-selection-weight 0.05 \
  --stability-selection-weight 0.05 \
  --nonpositive-selection-weight 0.02 \
  --folds 3 \
  --beta-grid 0.40,0.50,0.60 \
  --gamma-grid 0.25,0.50,0.75,1.00 \
  --zero-shrinkage-grid 0.02,0.05,0.10 \
  --bootstrap-samples 2000
