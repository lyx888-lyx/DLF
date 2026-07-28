#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Diagnostic-safe V8 run:
# - Stage A prediction only
# - Stage B frozen encoder/predictor, error head only
# - no shared-encoder joint training
# - deterministic Valid/Test evaluation
python3 train_unimodal_experts_v8.py \
  --dataset mosi \
  --gpu 0 \
  --config ./config/config.json \
  --save-root ./result/unimodal_experts_v8 \
  --modalities text audio vision \
  --seeds 1111 \
  --batch-size 16 \
  --num-workers 1 \
  --text-hidden-dim 256 \
  --audio-hidden-dim 96 \
  --vision-hidden-dim 96 \
  --text-layers 0 \
  --audio-layers 2 \
  --vision-layers 2 \
  --num-heads 4 \
  --ffn-multiplier 4 \
  --text-dropout 0.20 \
  --audio-dropout 0.35 \
  --vision-dropout 0.35 \
  --text-pooling cls \
  --layer-fusion final \
  --prediction-loss-text mae \
  --prediction-loss-av mse \
  --prediction-epochs-text 20 \
  --prediction-epochs-av 20 \
  --uncertainty-epochs 8 \
  --joint-mode disabled \
  --joint-epochs 0 \
  --prediction-patience-text 8 \
  --prediction-patience-av 8 \
  --learning-rate 1e-4 \
  --text-learning-rate 1e-5 \
  --uncertainty-learning-rate 3e-4 \
  --weight-decay 1e-2 \
  --uncertainty-weight 0.02 \
  --grad-clip 1.0
