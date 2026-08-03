#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-result}"
OUTPUT_DIR="${AVSC_OUTPUT_DIR:-${RESULT_ROOT}/missing_baseline/av_synergy_contribution_audit_v1/mosi}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-5000}"

python3 smoke_test_av_synergy_contribution.py

python3 analyze_av_synergy_contribution.py \
  --result-root "${RESULT_ROOT}" \
  --dataset mosi \
  --seeds 1111 1112 1113 1114 1115 \
  --splits valid test \
  --decision-split valid \
  --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}" \
  --output-dir "${OUTPUT_DIR}"

python3 audit_av_synergy_contribution.py \
  --result-dir "${OUTPUT_DIR}"

echo "AV synergy contribution audit complete: ${OUTPUT_DIR}"
