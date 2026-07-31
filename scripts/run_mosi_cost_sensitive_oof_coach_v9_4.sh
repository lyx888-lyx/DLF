#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-1111}"
INNER_FOLDS="${INNER_FOLDS:-3}"
INNER_VALID_FRACTION="${INNER_VALID_FRACTION:-0.20}"
SAVE_ROOT="${SAVE_ROOT:-./result/cost_sensitive_oof_coach_v94}"
TOP_OOF_CACHE="${TOP_OOF_CACHE:-./result/oof_tail_residual_experts_v92/mosi/seed_${SEED}/oof_cfcompat/nested_grouped_oof_cfcompat_cache_v92.pth}"

python3 build_nested_oof_expert_pool_v9_4.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --save-root "${SAVE_ROOT}" \
  --top-oof-cache "${TOP_OOF_CACHE}" \
  --inner-folds "${INNER_FOLDS}" \
  --inner-valid-fraction "${INNER_VALID_FRACTION}"

python3 train_cost_sensitive_oof_coach_v9_4.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --save-root "${SAVE_ROOT}"

python3 scripts/audit_cost_sensitive_oof_coach_v9_4.py \
  --root "${SAVE_ROOT}/mosi/seed_${SEED}"
