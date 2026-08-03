#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
V919_ROOT="${V919_ROOT:-result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}}"
V933_OUTPUT_DIR="${V933_OUTPUT_DIR:-${V919_ROOT}/v933_hierarchical_temporal_cfcompat}"
OUTER_FOLDS="${OUTER_FOLDS:-5}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-3}"
TEMPORAL_HIDDEN_DIM="${TEMPORAL_HIDDEN_DIM:-64}"
CONTEXT_HIDDEN_DIM="${CONTEXT_HIDDEN_DIM:-96}"
BRANCH_DROPOUT="${BRANCH_DROPOUT:-0.10}"
TEMPORAL_KERNEL_SIZE="${TEMPORAL_KERNEL_SIZE:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-20}"
EARLY_STOP="${EARLY_STOP:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
WRONG_CONTEXT_PRESERVATION_WEIGHT="${WRONG_CONTEXT_PRESERVATION_WEIGHT:-0.10}"
MISSING_TEMPORAL_WEIGHT="${MISSING_TEMPORAL_WEIGHT:-0.10}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
COMPAT="scripts/v925_scipy_compat_runner.py"

python3 "${COMPAT}" scripts/smoke_test_hierarchical_temporal_cfcompat_v9_33.py

python3 "${COMPAT}" analyze_hierarchical_temporal_cfcompat_v9_33.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v919-root "${V919_ROOT}" \
  --output-dir "${V933_OUTPUT_DIR}" \
  --outer-folds "${OUTER_FOLDS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --temporal-hidden-dim "${TEMPORAL_HIDDEN_DIM}" \
  --context-hidden-dim "${CONTEXT_HIDDEN_DIM}" \
  --branch-dropout "${BRANCH_DROPOUT}" \
  --temporal-kernel-size "${TEMPORAL_KERNEL_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-epochs "${MAX_EPOCHS}" \
  --early-stop "${EARLY_STOP}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --grad-clip-norm "${GRAD_CLIP_NORM}" \
  --wrong-context-preservation-weight "${WRONG_CONTEXT_PRESERVATION_WEIGHT}" \
  --missing-temporal-weight "${MISSING_TEMPORAL_WEIGHT}" \
  --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"

python3 "${COMPAT}" scripts/audit_hierarchical_temporal_cfcompat_v9_33.py \
  --result-dir "${V933_OUTPUT_DIR}"
