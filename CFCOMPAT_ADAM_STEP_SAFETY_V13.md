# CFCompatKD v13 — Train-Only Actual-Adam-Step Functional Safety

## Goal

v13 is a **metric-bearing candidate**, not another diagnostic-only experiment.  The end criterion is improvement on the frozen development protocol while preserving Test isolation.

The method is motivated by v12.3: stochastic optimizer windows frequently became first-order harmful to Teacher-beneficial and Frozen-S0-beneficial OOF groups, and Adam introduced additional harmful effective steps, while standalone nonlinear finite-step harm was rare.  v13 therefore moves the safety constraint from raw-gradient space to the parameter displacement that Adam actually proposes.

## What remains unchanged from v12

- Frozen S0 and the exact v8 residual-head-bank architecture.
- Five deterministic whole-video Train folds.
- Same shuffled Train loader and explicit missing-mode RNG.
- Exact v4 supervised-missing + DISTILL + 0.25×PRESERVE objective.
- Exact numerically stable v12 asymmetric raw-gradient surgery.
- Adam optimizer, learning rate, scheduler, and `update_epochs=10`.
- Train-video-holdout checkpoint selection: earliest epoch within 1% of absolute-best holdout J.
- 4-of-5 same-sign median residual consensus at inference.
- Official Valid first used only after all five fold banks are frozen.
- Official Test never constructed or accessed.

## Fixed Train-only sentinel groups

For each fold, sentinel events are selected **only from that fold's Train 4/5 videos** before training.  A separate deterministic `shuffle=False` loader is used so sentinel construction cannot consume the formal Train sampler RNG.

For every missing mode LA/LV/L, two historical definitions are reused without introducing a new selection threshold:

1. **Teacher-beneficial**: `baseline_error - teacher_error >= 0.02`.
2. **Frozen-S0-beneficial**: `baseline_error - s0_error >= 0.02`.

Frozen S0 residual features, S0 predictions, labels, and masks are cached once.  Valid/OOF-holdout/Test events never enter a sentinel cache.

## Safety loss

At the current residual parameters, each `(group, mode)` sentinel defines a mean MAE to the true Train label:

`L_safe = mean |S0_prediction + residual_delta - label|`.

The safety gradient is recomputed at every optimizer window through only the small residual bank; the frozen backbone is not rerun.

## Actual-step projection

The v12 raw update is first assigned normally.  Adam then updates its first/second moments and proposes the real displacement:

`delta_proposed = theta_after_Adam - theta_before`.

For each missing mode, v13 requires both first-order safety conditions:

`<delta, grad L_teacher_beneficial> <= 0`

`<delta, grad L_s0_beneficial> <= 0`.

If the proposed displacement violates either condition, v13 replaces it by the closest Euclidean displacement satisfying both halfspaces.  With two constraints this is solved by a deterministic active-set projection.  Different missing modes use disjoint residual heads, so the per-mode projections do not compete for parameters.

Adam's internal moment state is **not** projected or reset.  This intentionally preserves the repair history that v12.3 showed was useful for Teacher-nonbeneficial samples; every future Adam proposal is simply safety-checked again before it reaches the parameters.

There is no safety coefficient, epoch switch, warmup threshold, or Valid-fitted calibration.

## Frozen metric decision rules

### Direct improvement over v12

Before v13 Valid results are observed, v13 is required to satisfy all four:

- candidate J is strictly lower than v12 J;
- Teacher-beneficial NTR is strictly lower than v12;
- overall NTR is strictly lower than v12;
- Teacher-nonbeneficial NTR degrades by at most 3 percentage points versus v12.

The 3pp retention tolerance reuses the existing noninferiority convention; it is not tuned from v13.

### Legacy target gate

The original v12 gate remains unchanged:

- `J(v13) - J(v8) <= 0.002`;
- Teacher-beneficial NTR degradation versus v8 `<= 0.03`;
- Teacher-nonbeneficial NTR reduction versus v8 `>= 0.05`;
- overall NTR degradation versus v4 `<= 0.01`.

A full promotion verdict requires **both** the direct-v12 improvement gate and the unchanged legacy target gate.  Passing only the direct gate is recorded as real progress but not final promotion.

## Outputs to inspect

- `adam_step_safety_v13_valid_screen_summary.json`
- `adam_step_safety_v13_transfer_summary.csv`
- `adam_step_safety_v13_fold_manifest.csv`
- `adam_step_safety_v13_actual_step_safety_windows.csv`
- `adam_step_safety_v13_sentinel_manifest.csv`
- `adam_step_safety_v13_fold_epoch_metrics.csv`

The actual-step window diagnostics report how often v13 intervenes, how much of Adam's proposed displacement is removed, and the raw-surgery/Adam-effective cosine.  Those are mechanism diagnostics only; model promotion is decided by the frozen Valid metric gates above.
