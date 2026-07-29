#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python3 -m py_compile \
  trains/singleTask/model/OOFPositiveResidual_DLF.py \
  trains/singleTask/oof_positive_residual_system_v94.py \
  train_oof_positive_residual_v9_4.py \
  scripts/finalize_oof_positive_residual_v94.py \
  scripts/audit_oof_positive_residual_v94.py \
  scripts/smoke_oof_positive_residual_v94.py

bash -n scripts/run_mosi_oof_positive_residual_v94.sh
python3 scripts/smoke_oof_positive_residual_v94.py

echo "V9.4 syntax and utility checks passed"
