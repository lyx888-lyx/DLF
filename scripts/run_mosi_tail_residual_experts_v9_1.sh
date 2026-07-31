#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
TEACHER_GLOB="${TEACHER_GLOB:-/code/DLF/pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed*/DLF_mosi_seed*_best_valid.pth}"
SAVE_ROOT="${SAVE_ROOT:-./result/tail_residual_experts_v91}"

python3 train_tail_residual_experts_v9_1.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --save-root "${SAVE_ROOT}" \
  --teacher-glob "${TEACHER_GLOB}"

python3 scripts/audit_tail_residual_experts_v9_1.py \
  --root "${SAVE_ROOT}/mosi/seed_${SEED}"
