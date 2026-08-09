# CFCompatKD v12.1 — Frozen Post-Surgery Train-OOF Audit

## Purpose

v12 showed a partial intervention signal: asymmetric gradient surgery reduced
Teacher-beneficial harm relative to v10 while largely preserving
Teacher-nonbeneficial NTR repair, but it did not recover v8-level Frozen-S0
safety and failed the frozen promotion gate.

v12.1 does **not** train another model.  It asks the narrower mechanism
question left open by v12:

> After the supervised gradient has been made non-conflicting with the
> selective DISTILL/PRESERVE gradient, is the projected supervised direction
> (or the resulting surgery update) still harmful to held-out Train beneficial
> groups?

If yes, the selective gradient is not a sufficient proxy for functional safety.
If no, yet frozen v12 remains unsafe on Official Valid, then local
final-checkpoint gradient geometry is not sufficient to explain the trajectory
and nonlinear/trajectory-level drift becomes the stronger hypothesis.

## Frozen inputs

The audit loads the five already-trained v12 conservative residual-bank
checkpoints.  It does not select a checkpoint and it performs no optimizer
steps.

The video-grouped 5-fold assignment is the frozen v10/v12 assignment.  Each
MOSI Train sample is audited only by the fold for which its full video was held
out from residual training.

Official Test is never constructed or accessed.  Official Valid is not used in
any v12.1 statistic or decision.  The legacy v4 asset loader may materialize an
immutable Valid reference as an implementation prerequisite, exactly as in the
v11 diagnostic.

## Same local audit geometry as v11

For comparability with v11, each frozen fold checkpoint uses exhaustive,
equal-weight LA/LV/L missing views to estimate the expected residual-head Train
gradients:

```text
g_sup = SUPERVISED_ALL
g_sel = DISTILL_KD + 0.25 * PRESERVE
```

Held-out Train events provide OOF MAE gradients for:

- OOF Teacher-beneficial;
- OOF Teacher-nonbeneficial;
- OOF S0-beneficial;
- OOF S0-nonbeneficial;
- OOF all events.

A positive dot product between a Train direction and an OOF loss gradient means
an infinitesimal gradient-descent step along the Train direction is predicted
to lower OOF MAE.  A negative dot predicts local OOF harm.

## Analytic post-surgery replay

At each frozen checkpoint and for each audit mode, v12.1 replays exactly the v12
asymmetric projection in flattened float64 gradient space:

```text
if dot(g_sup, g_sel) < 0:
    g_sup_projected = g_sup - dot(g_sup, g_sel) / ||g_sel||^2 * g_sel
else:
    g_sup_projected = g_sup

g_removed = g_sup - g_sup_projected
g_update = g_sup_projected + g_sel
```

The audit records first-order OOF alignment for:

- `SUPERVISED_ALL_RAW`;
- `SELECTIVE_ONLY`;
- `TOTAL_RESIDUAL_OBJECTIVE_RAW`;
- `SUPERVISED_PROJECTED`;
- `SUPERVISED_REMOVED_CONFLICT`;
- `SURGERY_UPDATE`.

This is deliberately a **local frozen-checkpoint replay**.  It does not claim to
reconstruct historical Adam states or the exact per-window gradient vectors
that existed earlier in v12 training.

## Interpretation fixed before running

The main diagnostic is the ALL-mode matrix.

1. If `SUPERVISED_PROJECTED -> OOF_TEACHER_BENEFICIAL` remains harmful in 5/5
   folds, then removing direct conflict with `g_sel` is insufficient and
   `g_sel` is not a sufficient functional-safety anchor.
2. If `SURGERY_UPDATE -> OOF_TEACHER_BENEFICIAL` remains harmful in 5/5 folds,
   the complete local v12 update direction itself remains systematically unsafe
   for that group at the frozen checkpoints.
3. The same two checks are repeated for OOF S0-beneficial events because v8's
   strongest safety property was preservation of Frozen-S0-good behavior.
4. If the projected/update directions improve beneficial OOF groups in 5/5
   folds despite the known unsafe frozen-v12 Valid outcome, then the residual
   failure is more consistent with trajectory/nonlinear function drift than
   with a remaining local final-checkpoint gradient conflict.

These are diagnostic interpretations, not a model-promotion gate.  No threshold
or projection strength is tuned after observing v12.1.

## Outputs

The audit writes:

- `post_surgery_audit_v12p1_summary.json`
- `post_surgery_audit_v12p1_headline_gradient_matrix.csv`
- `post_surgery_audit_v12p1_gradient_influence_by_fold.csv`
- `post_surgery_audit_v12p1_gradient_influence_summary.csv`
- `post_surgery_audit_v12p1_projection_geometry.csv`
- `post_surgery_audit_v12p1_train_oof_events.csv`
- `post_surgery_audit_v12p1_event_summary.csv`
- `post_surgery_audit_v12p1_fold_manifest.csv`

The audit is diagnostic only: no new model is trained, no checkpoint is
selected, and Official Test remains untouched.
