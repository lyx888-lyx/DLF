#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 evaluate_modality_ablation_v7_5.py \
  --dataset mosi \
  --seed 1111 \
  --gpu 0 \
  --source-run-dir ./result/complementarity_v71/mosi/seed_1111 \
  --save-root ./result/modality_ablation_v75 \
  --batch-size 32 \
  --num-workers 1 \
  --committee-steps 600 \
  --conditions t a v ta tv av tav
