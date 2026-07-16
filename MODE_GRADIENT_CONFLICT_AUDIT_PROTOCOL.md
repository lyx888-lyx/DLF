# Stage 6A: Missing-Pattern Gradient Interference Audit

- Base: `experiment/cf-compat-kd-multiseed-v1` at `276fc26e6e7d1998f7dd746edfca1efbbb691e55`.
- Branch: `analysis/mode-gradient-conflict-v1`.
- Scope: MOSI seed1111 train split only, shuffle false, drop-last false, batch size unchanged.
- States: Gate3 initialization and CFCompatKD validation-best Student.
- Views: deterministic paired LAV, LA, LV, and L for every train sample.
- Task gradients: original complete DLF loss for LAV and frozen Stage 3 five-head task-only loss for LA/LV/L.
- KD gradients: frozen Stage 3 `sum(C * SmoothL1) / (sum(C) + 1e-8)` for each missing mode.
- Safety: eval mode, no optimizer, no backward accumulation, no step, no checkpoint creation, no valid/test access, exact RNG/parameter/buffer preservation.
- Statistics: batch-level semantic-group cosines, 4x4 first-order transfer, task/KD alignment, per-mode train compatibility quartiles, norm dominance, and 2000 fixed-seed bootstrap resamples.
- Representation: the 300-dimensional input to `backbone.proj1`, which the real DLF forward defines as the last shared fused representation before the final prediction head.

This stage emits diagnostic G1--G5 flags only. It does not create Stage 6B or implement an Adapter, mode-specific head, gradient surgery, or any training method.
