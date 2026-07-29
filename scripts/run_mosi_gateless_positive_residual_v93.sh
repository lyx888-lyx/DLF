#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
RUN_DIR="${RUN_DIR:-./result/complementarity_v71/mosi/seed_${SEED}}"
SAVE_ROOT="${SAVE_ROOT:-./result/gateless_positive_residual_v93}"
OUTPUT_DIR="$SAVE_ROOT/mosi/seed_${SEED}"
CACHE_PATH="$RUN_DIR/complementarity_v71_teacher_cache.pth"

if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[V9.3] Missing teacher cache; rebuilding it from the frozen V7.1 run..."
  python3 scripts/rebuild_v71_teacher_cache.py \
    --dataset mosi \
    --seed "$SEED" \
    --gpu "$GPU" \
    --run-dir "$RUN_DIR" \
    --batch-size 32 \
    --num-workers 1
fi

# V9.3 diagnostic:
# - remove the collapsible gate;
# - train bounded positive correction magnitude directly;
# - keep no-harm and over-correction penalties deliberately weak;
# - select the residual checkpoint by residual prediction quality on Valid;
# - only then search beta/shrinkage/gamma once on Valid.
python3 train_gateless_positive_residual_v9_3.py \
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
  --initial-magnitude-fraction 0.02 \
  --strong-threshold 1.0 \
  --neutral-radius 1e-6 \
  --region-weight-mode inverse_sqrt \
  --region-weight-min 0.50 \
  --region-weight-max 2.00 \
  --ordinary-positive-boost 1.50 \
  --huber-delta 0.30 \
  --residual-margin 0.05 \
  --no-harm-weight 0.05 \
  --overcorrection-weight 0.05 \
  --positive-target-weight 1.50 \
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
  --gamma-grid 0.05,0.10,0.20,0.30,0.50,0.75,1.00 \
  --zero-shrinkage-grid 0.02,0.05,0.10 \
  --bootstrap-samples 2000

python3 scripts/finalize_gateless_positive_residual_v93.py \
  --run-dir "$OUTPUT_DIR"
