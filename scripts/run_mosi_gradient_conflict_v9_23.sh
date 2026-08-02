#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-1111}"
GPU="${GPU:-0}"
ROOT="result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}"

python3 analyze_gradient_conflict_v9_23.py \
  --dataset mosi \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --v919-root "${ROOT}" \
  --outer-folds 5 \
  --batch-size 16 \
  --parameter-scope fusion_tail \
  --common-cosine-margin 0.02 \
  --minimum-mgda-norm 0.05 \
  --required-feasible-outer-folds 4

python3 scripts/audit_gradient_conflict_v9_23.py \
  --v919-root "${ROOT}" \
  --outer-folds 5
