#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"

python3 train_selective_category_coach_v9_5.py \
  --dataset mosi \
  --seed "$SEED" \
  --gpu "$GPU"

python3 scripts/audit_selective_category_coach_v9_5.py \
  --root "result/selective_category_coach_v95/mosi/seed_${SEED}"
