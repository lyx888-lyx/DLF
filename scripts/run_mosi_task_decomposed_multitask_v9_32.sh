#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
V919_ROOT="${V919_ROOT:-result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}}"
V932_OUTPUT_DIR="${V932_OUTPUT_DIR:-${V919_ROOT}/v932_task_decomposed_multitask}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-12}"
EARLY_STOP="${EARLY_STOP:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-1}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-fusion_tail}"
AUXILIARY_HIDDEN_DIM="${AUXILIARY_HIDDEN_DIM:-128}"
AUXILIARY_DROPOUT="${AUXILIARY_DROPOUT:-0.15}"
ORDINAL_WEIGHT="${ORDINAL_WEIGHT:-0.20}"
INTENSITY_WEIGHT="${INTENSITY_WEIGHT:-0.10}"
ORDINAL_MONOTONIC_WEIGHT="${ORDINAL_MONOTONIC_WEIGHT:-0.05}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
COMPAT="scripts/v925_scipy_compat_runner.py"

python3 "${COMPAT}" scripts/smoke_test_task_decomposed_multitask_v9_32.py

python3 "${COMPAT}" analyze_task_decomposed_multitask_v9_32.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v919-root "${V919_ROOT}" \
  --output-dir "${V932_OUTPUT_DIR}" \
  --outer-folds "${OUTER_FOLDS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-epochs "${MAX_EPOCHS}" \
  --early-stop "${EARLY_STOP}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --update-epochs "${UPDATE_EPOCHS}" \
  --trainable-scope "${TRAINABLE_SCOPE}" \
  --auxiliary-hidden-dim "${AUXILIARY_HIDDEN_DIM}" \
  --auxiliary-dropout "${AUXILIARY_DROPOUT}" \
  --ordinal-weight "${ORDINAL_WEIGHT}" \
  --intensity-weight "${INTENSITY_WEIGHT}" \
  --ordinal-monotonic-weight "${ORDINAL_MONOTONIC_WEIGHT}" \
  --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"

python3 "${COMPAT}" scripts/audit_task_decomposed_multitask_v9_32.py \
  --result-dir "${V932_OUTPUT_DIR}"
