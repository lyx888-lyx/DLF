#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RESULT_ROOT="${RESULT_ROOT:-result}"
SOURCE_DIR="${SOURCE_DIR:-${RESULT_ROOT}/missing_baseline/dlf_role_specialization_audit_v1/mosi/valid_train_audit}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/dlf_tail_risk_coupling_audit_v1/mosi/valid_only"

python3 smoke_test_dlf_tail_risk_coupling.py

python3 analyze_dlf_tail_risk_coupling.py \
  --result-root "${RESULT_ROOT}" \
  --source-dir "${SOURCE_DIR}" \
  --bootstrap-replicates 2000

python3 audit_dlf_tail_risk_coupling.py \
  --result-dir "${OUTPUT_DIR}"

echo "DLF tail-risk coupling audit complete: ${OUTPUT_DIR}"
