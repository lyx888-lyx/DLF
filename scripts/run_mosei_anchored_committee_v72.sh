#!/usr/bin/env bash
set -euo pipefail

cd /code/DLF

python3 evaluate_anchored_committee_v7_2.py \
  --dataset mosei \
  --seed 1111 \
  --gpu 0 \
  --batch-size 32 \
  --save-root ./result/anchored_committee_v72 \
  --teacher-checkpoint /code/DLF-mosei-generalization-v1/result/missing_baseline/mosei_generalization_v1/cfcompat/seed1111/DLF_mosei_seed1111_best_valid.pth \
  --teacher-checkpoint /code/DLF-mosei-generalization-v1/result/missing_baseline/mosei_generalization_v1/cfcompat/seed1112/DLF_mosei_seed1112_best_valid.pth \
  --teacher-checkpoint /code/DLF-mosei-generalization-v1/result/missing_baseline/mosei_generalization_v1/cfcompat/seed1113/DLF_mosei_seed1113_best_valid.pth \
  --teacher-checkpoint /code/DLF-mosei-generalization-v1/result/missing_baseline/mosei_generalization_v1/cfcompat/seed1114/DLF_mosei_seed1114_best_valid.pth \
  --teacher-checkpoint /code/DLF-mosei-generalization-v1/result/missing_baseline/mosei_generalization_v1/cfcompat/seed1115/DLF_mosei_seed1115_best_valid.pth
