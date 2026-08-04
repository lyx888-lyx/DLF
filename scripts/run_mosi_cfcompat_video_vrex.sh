#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
RESULT_ROOT="${RESULT_ROOT:-result}"
MODEL_SAVE_DIR="${MODEL_SAVE_DIR:-pt}"
OUTPUT_DIR="${RESULT_ROOT}/missing_baseline/cfcompat_video_vrex_v1/mosi/seed${SEED}"

python3 smoke_test_cfcompat_video_vrex.py

python3 audit_video_domains.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu-ids "${GPU}" \
  --result-root "${RESULT_ROOT}" \
  --samples-per-video 4

TRAIN_ARGS=(
  --dataset mosi
  --seed "${SEED}"
  --gpu-ids "${GPU}"
  --num-workers 1
  --result-root "${RESULT_ROOT}"
  --model-save-dir "${MODEL_SAVE_DIR}"
  --samples-per-video 4
)

if [[ "${SEED}" == "1114" ]]; then
  : "${FIXED_LAMBDA:?Set FIXED_LAMBDA to the lambda selected by seed1111}"
  TRAIN_ARGS+=(--fixed-lambda "${FIXED_LAMBDA}")
fi

python3 run_cfcompat_video_vrex_locked.py "${TRAIN_ARGS[@]}"

python3 audit_cfcompat_video_vrex.py \
  --result-dir "${OUTPUT_DIR}"

echo "CFCompat Video-VREx run complete: ${OUTPUT_DIR}"
