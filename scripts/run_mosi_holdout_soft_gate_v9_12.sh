#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
OUTER_FOLD="${OUTER_FOLD:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_holdout_soft_gate_v9_12.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --outer-fold "${OUTER_FOLD}" \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_holdout_soft_gate_v9_12.py \
  --root "result/holdout_soft_gate_v912/mosi/seed_${SEED}"
