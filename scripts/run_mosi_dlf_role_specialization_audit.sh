#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/dlf_role_specialization_audit_v1/mosi/valid_train_audit"

python3 smoke_test_dlf_role_specialization.py

python3 analyze_dlf_role_specialization.py \
  --probe-dataset mosi \
  --distribution-datasets mosi mosei \
  --seeds 1111 1114 \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}"

python3 audit_dlf_role_specialization_v2.py \
  --result-dir "${OUTPUT_DIR}"

echo "DLF role-specialization audit complete: ${OUTPUT_DIR}"
