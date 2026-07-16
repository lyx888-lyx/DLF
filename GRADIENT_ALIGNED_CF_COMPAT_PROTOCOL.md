# Stage 6B: Gradient-Aligned Counterfactual Compatibility Distillation

- Base: `analysis/mode-gradient-conflict-v1` at `13a6eb7d7e9708e6480033f80be7f48312739c04`.
- Branch: `feature/gradient-aligned-cfcompat-v1`.
- Track: DLF benchmark/original protocol, MOSI seed1111 only.
- Frozen forward definition: the Gate 3 Full Teacher, independently initialized
  `MissingModalityWrapper(DLF)` Student, Stage 3 train-only compatibility cache,
  weighted-normalized SmoothL1 KD, task losses, missing RNG, Adam optimizer,
  scheduler, update accumulation, and validation-only main selection are reused.
- Supervised anchor: `L_sup = L_full + L_missing`; the task gradient is never
  projected, scaled, clipped, or otherwise modified.
- `manual_replay`: replay `grad(L_sup) + grad(L_kd)` and require real DLF
  single-batch gradient, single-step parameter, optimizer, and scheduler
  equivalence before any training.
- Real DLF preflight showed that separately materializing and adding gradients
  has optimizer-visible FP32 reduction-order round-off despite global cosine
  above 0.999999.  More importantly, ten such additions do not reproduce the
  original `AccumulateGrad` ordering.  Each training batch therefore executes
  exactly one native Stage 3 total-loss backward as its numerical baseline.
  Manual Replay leaves that gradient untouched; the two interventions add only
  `g_kd_used - g_kd_raw`.  This is algebraically `g_sup + g_kd_used`, preserves
  the original ten-batch accumulation exactly, and introduces no extra loss.
- `conflict_drop`: use the unchanged KD gradient when the global FP32 dot product
  is non-negative and zero it exactly when the dot product is negative.
- `task_anchored_projection`: on a negative global FP32 dot product only, use
  `g_kd - dot(g_sup,g_kd)/(||g_sup||^2+1e-12) * g_sup`.  There is no symmetric,
  layer-wise, parameter-wise, mode-wise, or norm-restoring operation.
- The historical Stage 3 loop accumulates ten batches before each optimizer
  update (`update_epochs=10`).  Stage 6B processes each batch before that frozen
  accumulation boundary and performs exactly one optimizer step per original
  boundary.  Stage 3 did not invoke gradient clipping, so Stage 6B adds none.
- Main checkpoints are selected only by validation J.  Test-best checkpoints are
  isolated and explicitly diagnostic.  Evaluation loads only the Student and
  does not load the Teacher, evaluator, or compatibility cache.
- Online outputs include step, epoch, parameter-group, fixed train-only probe,
  actual optimizer-delta, and train-only representation diagnostics.  These
  diagnostics never alter the loss or update.
- Formal Conflict-Drop and GA-CFCompatKD runs are blocked until Manual Replay
  reproduces epoch 9 and the locked J/LAV/MissingMacro results within `1e-4`.
