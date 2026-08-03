#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ROOT="${V919_ROOT:-result/full_same_stack_nested_crossfit_v919/mosi/seed_1111}"
OUTPUT_DIR="${V929_OUTPUT_DIR:-${ROOT}/v929_expert_pool_viability_audit}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
TARGET_MAE="${TARGET_MAE:-0.70}"
SHRINKAGE_LAMBDA="${SHRINKAGE_LAMBDA:-0.01}"
COMPAT="scripts/v925_scipy_compat_runner.py"

python3 "${COMPAT}" scripts/smoke_test_expert_pool_viability_v9_29.py
python3 "${COMPAT}" analyze_expert_pool_viability_v9_29.py \
  --root "${ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --outer-folds "${OUTER_FOLDS}" \
  --target-mae "${TARGET_MAE}" \
  --shrinkage-lambda "${SHRINKAGE_LAMBDA}"
python3 "${COMPAT}" scripts/audit_expert_pool_viability_v9_29.py \
  --result-dir "${OUTPUT_DIR}"
