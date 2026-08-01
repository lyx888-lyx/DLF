#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_strict_oof_small_scale_distillation_v9_16.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_strict_oof_small_scale_distillation_v9_16.py \
  --root "result/strict_oof_small_scale_distillation_v916/mosi/seed_${SEED}"
