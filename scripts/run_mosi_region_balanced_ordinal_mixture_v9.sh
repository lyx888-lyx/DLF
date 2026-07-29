#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
RUN_DIR="${RUN_DIR:-./result/complementarity_v71/mosi/seed_${SEED}}"
SAVE_ROOT="${SAVE_ROOT:-./result/region_balanced_ordinal_mixture_v9}"
CACHE_PATH="$RUN_DIR/complementarity_v71_teacher_cache.pth"

if [[ ! -f "$CACHE_PATH" ]]; then
  echo "[V9] Missing teacher cache; rebuilding it from the frozen V7.1 run..."
  python3 scripts/rebuild_v71_teacher_cache.py \
    --dataset mosi \
    --seed "$SEED" \
    --gpu "$GPU" \
    --run-dir "$RUN_DIR" \
    --batch-size 32 \
    --num-workers 1
fi

# First diagnostic run: balanced Huber + polarity/ordinal soft mixture.
# GroupDRO is deliberately disabled. Enable it only as a later isolated ablation.
python3 train_region_balanced_ordinal_mixture_v9.py \
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
  --neutral-expert-range 0.75 \
  --max-mixture-blend 0.35 \
  --initial-mixture-blend 0.02 \
  --gate-temperature 1.0 \
  --strong-threshold 1.0 \
  --neutral-radius 1e-6 \
  --region-weight-mode inverse_sqrt \
  --region-weight-min 0.50 \
  --region-weight-max 2.00 \
  --huber-delta 0.50 \
  --group-dro-weight 0.0 \
  --max-epochs 16 \
  --early-stop 5 \
  --learning-rate 3e-4 \
  --weight-decay 1e-3 \
  --valid-mae-tolerance 0.001 \
  --fold-mae-tolerance 0.010 \
  --worst-region-tolerance 0.10 \
  --robust-selection-weight 0.05 \
  --ordinary-positive-selection-weight 0.02 \
  --stability-selection-weight 0.05 \
  --folds 3 \
  --beta-grid 0.40,0.50,0.60 \
  --alpha-grid 0.50,0.75,1.00 \
  --bootstrap-samples 2000
