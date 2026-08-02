#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-1111}"
ROOT="result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}"
COMPAT="scripts/v925_scipy_compat_runner.py"

python3 "${COMPAT}" scripts/smoke_test_support_stability_v9_26.py

python3 "${COMPAT}" analyze_support_stability_self_knowledge_v9_26.py \
  --v919-root "${ROOT}" \
  --outer-folds 5 \
  --shrinkage-lambda 0.01 \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed "${SEED}"

python3 "${COMPAT}" scripts/audit_support_stability_v9_26.py \
  --v919-root "${ROOT}" \
  --outer-folds 5
