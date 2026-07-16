# Stage 5A: Compatibility-Routed Dual-Teacher Distillation

- Base branch: `experiment/cf-compat-kd-multiseed-v1`
- Locked base commit: `276fc26e6e7d1998f7dd746edfca1efbbb691e55`
- Implementation branch: `feature/compat-routed-dual-teacher-v1`
- Dataset/seed: MOSI / 1111, benchmark original protocol.
- Full Teacher and Student initialization: locked Gate 3 validation-best checkpoint.
- Mode Teacher: locked Stage 1 ModDrop validation-best predictions from the Stage 3 train-only cache.

For every training sample, the sole auxiliary budget is
`alpha * d_full + (1 - alpha) * d_mode`, followed by one batch mean. The fixed
routes are `mode_only` (alpha=0), `uniform_dual` (alpha=0.5), and
`compatibility_routed` (alpha=the frozen Stage 3 compatibility). There is no
separate normalization, residual, reliability term, trainable gate, feature KD,
or inference-time Teacher.

The train-only suitability audit covers exactly 1284 unique samples, preserves
all RNG and model modes, and never opens valid/test. Formal runs use online
frozen Full-Teacher LAV forward passes and cache-bound LA/LV/L Mode targets.
Validation/test evaluation is Student-only. Main checkpoints are selected only
by validation J; test-best checkpoints and predictions are diagnostic-only.
