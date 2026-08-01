#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"

python3 train_distributional_target_nearest_expert_v9_6.py \
  --dataset mosi \
  --gpu "${GPU}" \
  --seed "${SEED}"

python3 scripts/audit_distributional_target_nearest_expert_v9_6.py \
  --root "result/distributional_target_nearest_expert_v96/mosi/seed_${SEED}"
