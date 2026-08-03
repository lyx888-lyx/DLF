#!/usr/bin/env bash
set -euo pipefail

ROOT="${V919_ROOT:-result/full_same_stack_nested_crossfit_v919/mosi/seed_1111}"
OUTPUT_DIR="${V929_OUTPUT_DIR:-${ROOT}/v929_expert_pool_viability_audit}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
TARGET_MAE="${TARGET_MAE:-0.70}"
SHRINKAGE_LAMBDA="${SHRINKAGE_LAMBDA:-0.01}"

python scripts/smoke_test_expert_pool_viability_v9_29.py
python analyze_expert_pool_viability_v9_29.py \
  --root "${ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --outer-folds "${OUTER_FOLDS}" \
  --target-mae "${TARGET_MAE}" \
  --shrinkage-lambda "${SHRINKAGE_LAMBDA}"
python scripts/audit_expert_pool_viability_v9_29.py \
  --result-dir "${OUTPUT_DIR}"
