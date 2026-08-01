#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_attainable_frontier_coach_v9_7.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_attainable_frontier_coach_v9_7.py \
  --root "result/attainable_frontier_coach_v97/mosi/seed_${SEED}"
