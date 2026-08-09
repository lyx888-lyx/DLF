# CFCompatKD v12.3 — Window-Level Actual Adam-Step Train-OOF Audit

## Purpose

v12.2 showed that the expected full-Train post-surgery gradient is generally beneficial-safe even while beneficial OOF failures accumulate. v12.3 therefore moves one mechanism level down: from epoch-level expected gradients to the finite optimizer windows that actually changed the residual parameters.

v12.3 defines **no new candidate model** and no promotion rule. It is a Train-OOF mechanism diagnostic only.

## Frozen audit interval

Before running v12.3, the audited interval is fixed to **epochs 4 through 12 inclusive**. v12.2 already identified this interval as the rapid function-movement region: residual magnitude and label crossing increase sharply while beneficial NTR starts to accumulate.

Every optimizer update window in every one of these epochs is included. No window is selected from v12.3 results.

## Exact replay requirement

The script reuses the formal v12 fold trainer and the same numerical projection hotfix. During training, hooks only copy tensors already produced by the real update path. OOF diagnostics are deferred until all five folds have completed.

Before any window result is interpreted, every fold must reproduce formal v12 exactly on:

- conservative selected epoch;
- absolute-best Train-video-holdout epoch;
- missing-mode sequence SHA256;
- selected residual-bank state content hash;
- every selected-state tensor.

If any replay check fails, v12.3 stops without a mechanism conclusion.

## Captured window quantities

For each formal v12 optimizer window in epochs 4..12, v12.3 captures:

- accumulated supervised missing gradient `g_sup`;
- accumulated selective gradient `g_sel`;
- projected supervised gradient;
- post-surgery gradient `g_update`;
- residual parameters immediately before Adam step;
- residual parameters immediately after Adam step;
- actual displacement `delta_theta = theta_after - theta_before`.

The actual Adam effective gradient-like direction is defined only for comparison as `-delta_theta`.

## OOF evaluation

After exact replay is finished, each captured before/after parameter state is evaluated on the corresponding fold's held-out Train videos using exhaustive LA/LV/L views.

The fixed groups are:

- `OOF_TEACHER_BENEFICIAL`;
- `OOF_TEACHER_NONBENEFICIAL`;
- `OOF_S0_BENEFICIAL`;
- `OOF_S0_NONBENEFICIAL`.

For each window/group, v12.3 compares three levels.

### 1. Raw surgery first-order direction

For OOF gradient `g_oof`, `dot(g_update, g_oof) > 0` means a gradient-descent step on the raw surgery direction is locally improving. A negative dot is labeled `RAW_SURGERY_DIRECTION_HARM`.

### 2. Actual Adam first-order direction

The real finite parameter displacement is `delta_theta`. `dot(delta_theta, g_oof) < 0` means the actual Adam step is first-order improving. If raw surgery is safe but this dot is positive, the window is labeled `ADAM_TRANSFORM_HARM`.

### 3. Realized finite-step OOF change

The same OOF subgroup loss is measured at `theta_before` and `theta_after`. If raw surgery and Adam's actual displacement are both first-order safe but the realized OOF loss increases, the window is labeled `NONLINEAR_FINITE_STEP_HARM`.

Otherwise it is labeled `SAFE_OR_IMPROVING`.

These labels use only sign. No magnitude threshold is tuned from the audit.

## Interpretation

The intended diagnostic split is:

- many `RAW_SURGERY_DIRECTION_HARM` windows: expected epoch gradients were hiding harmful finite-window heterogeneity;
- raw-safe but many `ADAM_TRANSFORM_HARM` windows: Adam momentum / coordinate-wise preconditioning changes the effective direction materially;
- raw-safe and Adam-first-order-safe but many `NONLINEAR_FINITE_STEP_HARM` windows: curvature / finite-step nonlinear function change dominates;
- mostly `SAFE_OR_IMPROVING` windows despite accumulating failures: the next audit must move below the 10-mini-batch window, into within-window microbatch ordering/heterogeneity or longer-memory state effects.

## Protocol boundary

- Development seed: 1113.
- Same whole-video 5-fold Train split as formal v12.
- Same v12 residual architecture, loss, optimizer, update interval, missing RNG, selector, and numerical projection realization.
- No new model-selection gate.
- Official Valid is not used in v12.3 statistics or decisions; the legacy v4 loader may materialize the immutable Valid reference as an implementation prerequisite.
- Official Test is never constructed or accessed.
