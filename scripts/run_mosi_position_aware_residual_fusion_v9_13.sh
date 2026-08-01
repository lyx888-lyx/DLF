#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_position_aware_residual_fusion_v9_13.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_position_aware_residual_fusion_v9_13.py \
  --root "result/position_aware_residual_fusion_v913/mosi/seed_${SEED}"
