#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
NUM_WORKERS="${NUM_WORKERS:-1}"

python3 train_relative_regret_coach_v9_10_aligned.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --num-workers "${NUM_WORKERS}"

python3 scripts/audit_relative_regret_coach_v9_10.py \
  --root "result/relative_regret_coach_v910/mosi/seed_${SEED}"
