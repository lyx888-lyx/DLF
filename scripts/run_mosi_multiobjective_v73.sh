#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 evaluate_multiobjective_calibration_v7_3.py \
  --dataset mosi \
  --seed 1111 \
  --gpu 0 \
  --run-dir ./result/complementarity_v71/mosi/seed_1111 \
  --save-root ./result/multiobjective_calibration_v73 \
  --batch-size 32 \
  --beta-step 0.05 \
  --scale-min 0.90 \
  --scale-max 1.15 \
  --scale-step 0.01 \
  --bias-min -0.10 \
  --bias-max 0.10 \
  --bias-step 0.01 \
  --valid-mae-tolerance 0.001 \
  --fold-mae-tolerance 0.010 \
  --folds 3 \
  --bootstrap-samples 2000
