#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m py_compile \
  train_gateless_positive_residual_v9_3.py \
  trains/singleTask/model/GatelessPositiveResidual_DLF.py \
  trains/singleTask/gateless_positive_residual_system_v93.py \
  scripts/finalize_gateless_positive_residual_v93.py \
  scripts/audit_gateless_positive_residual_v93.py

bash -n scripts/run_mosi_gateless_positive_residual_v93.sh

echo "V9.3 syntax checks passed"
