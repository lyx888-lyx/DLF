#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-1111}"
ROOT="result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}"

python3 aggregate_same_stack_nested_crossfit_v9_19.py \
  --root "${ROOT}" \
  --outer-folds 5

python3 scripts/audit_same_stack_nested_crossfit_v9_19.py \
  --root "${ROOT}" \
  --aggregate
