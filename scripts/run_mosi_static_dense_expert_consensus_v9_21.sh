#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-1111}"
ROOT="result/full_same_stack_nested_crossfit_v919/mosi/seed_${SEED}"

python3 analyze_static_dense_expert_consensus_v9_21.py \
  --root "${ROOT}" \
  --outer-folds 5 \
  --shrinkage-lambda 0.01 \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed "${SEED}"

python3 scripts/audit_static_dense_expert_consensus_v9_21.py \
  --root "${ROOT}" \
  --outer-folds 5
