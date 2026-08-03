#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-1111}"
V930_DIR="${V930_DIR:-result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}/v930_observable_attribute_experts}"
V931_OUTPUT_DIR="${V931_OUTPUT_DIR:-${V930_DIR}/v931_attribute_gain_alignment}"
TOP_FRACTION="${TOP_FRACTION:-0.20}"
QUANTILE_BINS="${QUANTILE_BINS:-5}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-1131}"
COMPAT="scripts/v925_scipy_compat_runner.py"

python3 "${COMPAT}" scripts/smoke_test_observable_attribute_gain_v9_31.py

python3 "${COMPAT}" analyze_observable_attribute_gain_v9_31.py \
  --v930-dir "${V930_DIR}" \
  --output-dir "${V931_OUTPUT_DIR}" \
  --top-fraction "${TOP_FRACTION}" \
  --quantile-bins "${QUANTILE_BINS}" \
  --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}" \
  --min-spearman 0.05 \
  --min-win-auc 0.55 \
  --min-top-gain 0.005 \
  --min-top-bottom-lift 0.010 \
  --required-positive-folds 4

python3 "${COMPAT}" scripts/audit_observable_attribute_gain_v9_31.py \
  --result-dir "${V931_OUTPUT_DIR}"
