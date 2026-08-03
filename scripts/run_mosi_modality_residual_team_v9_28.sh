#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-120}"
ROOT="result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}"
COMPAT="scripts/v925_scipy_compat_runner.py"

python3 "${COMPAT}" scripts/smoke_test_modality_residual_team_v9_28.py

python3 "${COMPAT}" \
  analyze_modality_residual_team_v9_28.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v919-root "${ROOT}" \
  --outer-folds 5 \
  --temporal-bins 4 \
  --hidden-dim 64 \
  --dropout 0.15 \
  --correction-max 0.75 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate 0.002 \
  --weight-decay 0.001 \
  --correction-l1 0.01 \
  --coefficient-l2 0.01 \
  --coefficient-upper-bound 1.0 \
  --shrinkage-lambda 0.01 \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed "${SEED}"

python3 "${COMPAT}" scripts/audit_modality_residual_team_v9_28.py \
  --v919-root "${ROOT}" \
  --outer-folds 5
