# Stage 7A Modality Utility and Conditional Contribution Audit

This document freezes the implementation contract for MUCCA.

## Provenance and scope

- The analysis branch starts from `analysis/mode-gradient-conflict-v1` at
  `13a6eb7d7e9708e6480033f80be7f48312739c04`.
- The audit reads MOSI `train` and `valid` only. It never constructs or reads a
  test dataset.
- It compares seed1111 `gate3_init` with seed1111
  `cfcompat_best_valid`. Both checkpoint paths and SHA256 values must be resolved
  from Stage3 manifests, and the CFCompat path must be validation-selected.
- `gate3_init` loads the Gate3 validation-best DLF backbone and then constructs
  the Stage3 `MissingModalityWrapper`, leaving its missing tokens and mask
  adapter at their defined initialization.
- All models run in `eval()` mode. No optimizer, parameter update, buffer
  update, training checkpoint, new teacher, loss, adapter, or head is allowed.

## Fixed analysis

- For both train and valid, store sample-bound LAV/LA/LV/L predictions and MOSI
  regression metrics.
- Compute sample gain relative to L-only error for audio, vision, and audio plus
  vision.
- Create ten deterministic Sattolo derangements per split. Each mapping is a
  split-local, fixed-point-free bijection. The same mapping is used by both
  frozen states. Audio and vision use the same mapping within a repeat so the AV
  shuffle is a coherent sample-level counterfactual.
- MOSI is aligned and has no explicit audio/vision length fields in this
  configuration. The complete feature tensor is permuted. Its inferred
  non-padding mask and valid-step count are deterministic functions of the
  tensor and therefore move with it. Generic bound-field helpers and tests also
  verify explicit mask, length, and auxiliary fields move under one mapping.
- Compute audio/vision input gradients only with `torch.autograd.grad`.
  Non-zero feature rows define valid aligned positions; padding is excluded.
- Reuse exactly the Stage6A shared representation:
  `MissingModalityWrapper.backbone.proj1` forward-pre-hook input.
- Use paired bootstrap means with 2,000 resamples and fixed, content-derived
  seeds.

## Conditional analysis

- Labels use `[-3,-1)`, `[-1,0)`, `[0,1)`, and `[1,3]`.
- Audio uses `compat_LV`; vision uses `compat_LA`. Train values determine
  quartile edges, and valid uses those same edges. The locked Stage3 cache
  provides train compatibility. Valid evaluator deltas are mapped through the
  corresponding train empirical CDF; valid never refits ranks or quartiles.
- L-only absolute-error quartiles are fitted independently for each state on
  train and reused unchanged on valid.
- Every conditional table reports gain, shuffle damage, sensitivity, and shared
  representation shift.

## Registered decisions

- **A. Utility Supported** requires both the valid gain mean CI lower bound and
  valid shuffle-damage mean CI lower bound to be greater than zero.
- **B. Underuse Supported** requires neither CI to support a positive effect,
  mean shuffle prediction change below 10% of correct-LAV prediction standard
  deviation, and mean relative representation shift below 0.10.
- **C. Used but Unreliable** requires a prediction or representation change
  above those registered thresholds plus non-positive mean gain or no stable
  positive shuffle damage.
- All other outcomes are **D. Mixed / Inconclusive**.
- CFCompat enhancement or weakening requires paired positive or negative
  evidence for both gain and shuffle damage. Otherwise it is maintained only
  when both classifications match and both paired intervals include zero; all
  remaining cases are mixed.

## Integrity and reproduction

- Parameters, buffers, parameter `.grad` fields, and Python/NumPy/Torch/CUDA RNG
  states are checked before and after each state.
- Raw CSVs are written first. Every summary, decision, comparison, JSON summary,
  and Markdown report is then regenerated from those CSVs.
- `--summary-only --verify-existing-artifacts` verifies the old manifest,
  regenerates all derived outputs, and requires the resulting manifest to be
  byte-identical.
- Result artifacts are excluded from Git.
- Normal completion stops after the Stage7A report and does not create Stage 7B
  or implement a result-directed method.
