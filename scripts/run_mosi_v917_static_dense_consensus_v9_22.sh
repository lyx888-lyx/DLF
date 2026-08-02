#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-1111}"
ROOT="result/fixed_expert_region_audit_v917/mosi/seed_${SEED}"

python3 analyze_v917_static_dense_consensus_v9_22.py \
  --v917-root "${ROOT}" \
  --shrinkage-lambda 0.01 \
  --validation-group-folds 5 \
  --bootstrap-repetitions 2000 \
  --bootstrap-seed "${SEED}"

python3 scripts/audit_v917_static_dense_consensus_v9_22.py \
  --v917-root "${ROOT}"
