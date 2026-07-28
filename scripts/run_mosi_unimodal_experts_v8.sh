#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 train_unimodal_experts_v8.py \
  --dataset mosi \
  --gpu 0 \
  --config ./config/config.json \
  --save-root ./result/unimodal_experts_v8 \
  --modalities text audio vision \
  --seeds 1111 \
  --batch-size 32 \
  --num-workers 1 \
  --text-hidden-dim 128 \
  --audio-hidden-dim 96 \
  --vision-hidden-dim 96 \
  --text-layers 3 \
  --audio-layers 2 \
  --vision-layers 2 \
  --num-heads 4 \
  --ffn-multiplier 4 \
  --text-dropout 0.22 \
  --audio-dropout 0.35 \
  --vision-dropout 0.35 \
  --layer-fusion final \
  --prediction-epochs-text 15 \
  --prediction-epochs-av 20 \
  --uncertainty-epochs 6 \
  --joint-epochs 6 \
  --prediction-patience-text 6 \
  --prediction-patience-av 8 \
  --joint-patience 4 \
  --learning-rate 1e-4 \
  --text-learning-rate 2e-5 \
  --uncertainty-learning-rate 3e-4 \
  --weight-decay 1e-2 \
  --uncertainty-weight 0.10 \
  --grad-clip 1.0 \
  --max-prediction-degradation 0.005 \
  --max-corr-degradation 0.005
