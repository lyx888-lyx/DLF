#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_predicted_region_router_v9_11.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_predicted_region_router_v9_11.py \
  --root "result/predicted_region_router_v911/mosi/seed_${SEED}"
