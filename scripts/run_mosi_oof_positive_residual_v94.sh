#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
RUN_DIR="${RUN_DIR:-./result/complementarity_v71/mosi/seed_${SEED}}"
SAVE_ROOT="${SAVE_ROOT:-./result/oof_positive_residual_v94}"
OUTPUT_DIR="$SAVE_ROOT/mosi/seed_${SEED}"
CACHE_PATH="$RUN_DIR/complementarity_v71_teacher_cache.pth"

bash scripts/check_oof_positive_residual_v94.sh

if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[V9.4] Missing teacher cache; rebuilding it from the frozen V7.1 run..."
  python3 scripts/rebuild_v71_teacher_cache.py \
    --dataset mosi \
    --seed "$SEED" \
    --gpu "$GPU" \
    --run-dir "$RUN_DIR" \
    --batch-size 32 \
    --num-workers 1
fi

# V9.4 diagnostic:
# - grouped 5-fold cross-fitting at the calibration layer;
# - magnitude and activation are optimized in separate stages;
# - positive-class weighted focal BCE prevents the gate from choosing all-off;
# - residual is deployed only on its matching calibrated reference;
# - original V7.1 and the known zero-shrinkage policy remain legal fallbacks.
python3 train_oof_positive_residual_v9_4.py \
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
  --initial-gate-probability 0.35 \
  --initial-magnitude-fraction 0.02 \
  --strong-threshold 1.0 \
  --neutral-radius 1e-6 \
  --region-weight-mode inverse_sqrt \
  --region-weight-min 0.50 \
  --region-weight-max 2.00 \
  --ordinary-positive-boost 1.50 \
  --huber-delta 0.30 \
  --residual-margin 0.05 \
  --crossfit-folds 5 \
  --calibrator-l2-grid 0.1,1.0,10.0,100.0 \
  --calibrator-blend-grid 0.0,0.25,0.50,0.75,1.0 \
  --max-calibration-shift 0.75 \
  --magnitude-epochs 20 \
  --magnitude-early-stop 6 \
  --gate-epochs 15 \
  --gate-early-stop 5 \
  --learning-rate 3e-4 \
  --weight-decay 1e-3 \
  --magnitude-active-weight 1.50 \
  --magnitude-zero-weight 0.05 \
  --magnitude-over-weight 0.05 \
  --gate-focal-gamma 1.50 \
  --gate-pos-weight-cap 8.0 \
  --valid-mae-tolerance 0.001 \
  --fold-mae-tolerance 0.010 \
  --worst-region-tolerance 0.10 \
  --nonpositive-mae-tolerance 0.010 \
  --robust-selection-weight 0.05 \
  --stability-selection-weight 0.05 \
  --nonpositive-selection-weight 0.02 \
  --folds 3 \
  --beta-grid 0.40,0.50,0.60 \
  --gamma-grid 0.10,0.25,0.50,0.75,1.00 \
  --zero-shrinkage-grid 0.02,0.05,0.10 \
  --gate-threshold-grid 0.30,0.40,0.50,0.60,0.70 \
  --bootstrap-samples 2000

python3 scripts/finalize_oof_positive_residual_v94.py \
  --run-dir "$OUTPUT_DIR"
