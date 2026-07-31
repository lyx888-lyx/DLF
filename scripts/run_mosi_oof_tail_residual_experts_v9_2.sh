#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
OUTER_FOLDS="${OUTER_FOLDS:-3}"
INNER_VALID_FRACTION="${INNER_VALID_FRACTION:-0.15}"
TEACHER_GLOB="${TEACHER_GLOB:-/code/DLF/pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed*/DLF_mosi_seed*_best_valid.pth}"
SAVE_ROOT="${SAVE_ROOT:-./result/oof_tail_residual_experts_v92}"

python3 build_grouped_oof_cfcompat_v9_2.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --save-root "${SAVE_ROOT}" \
  --outer-folds "${OUTER_FOLDS}" \
  --inner-valid-fraction "${INNER_VALID_FRACTION}"

python3 train_oof_tail_residual_experts_v9_2.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --save-root "${SAVE_ROOT}" \
  --teacher-glob "${TEACHER_GLOB}"

python3 scripts/audit_oof_tail_residual_experts_v9_2.py \
  --root "${SAVE_ROOT}/mosi/seed_${SEED}"
