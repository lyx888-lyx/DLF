#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
V9_ROOT="${V9_ROOT:-./result/role_conditioned_experts_v9/mosi/seed_${SEED}}"
V92_ROOT="${V92_ROOT:-./result/oof_tail_residual_experts_v92/mosi/seed_${SEED}}"
SAVE_ROOT="${SAVE_ROOT:-./result/ordinal_advantage_coach_v93}"

python3 train_ordinal_advantage_coach_v9_3.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v9-root "${V9_ROOT}" \
  --v92-root "${V92_ROOT}" \
  --save-root "${SAVE_ROOT}"

python3 scripts/audit_ordinal_advantage_coach_v9_3.py \
  --root "${SAVE_ROOT}/mosi/seed_${SEED}"
