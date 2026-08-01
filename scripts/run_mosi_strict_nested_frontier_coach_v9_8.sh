#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_strict_nested_frontier_coach_v9_8.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_strict_nested_frontier_coach_v9_8.py \
  --root "result/strict_nested_frontier_coach_v98/mosi/seed_${SEED}"
