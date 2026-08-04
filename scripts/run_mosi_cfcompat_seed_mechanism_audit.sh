#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/cfcompat_seed_mechanism_audit_v1/mosi/valid_train_audit"

python3 smoke_test_cfcompat_seed_mechanisms.py

python3 run_cfcompat_seed_mechanisms_locked.py \
  --dataset mosi \
  --seeds 1111 1114 \
  --gpu-ids "${GPU}" \
  --num-workers 1 \
  --result-root "${RESULT_ROOT}" \
  --model-save-dir "${MODEL_SAVE_DIR}" \
  --probe-samples 64

python3 audit_cfcompat_seed_mechanisms.py \
  --result-dir "${OUTPUT_DIR}"

echo "CFCompatKD seed mechanism audit complete: ${OUTPUT_DIR}"
