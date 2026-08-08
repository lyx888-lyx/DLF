# CFCompatKD v11 — Train-OOF Objective / Gradient Audit

## Purpose

v11 is **diagnostic only**. It does not train a new model, tune a threshold, or select a checkpoint.
It loads the five frozen v10 conservative cross-fit residual banks and audits what the existing
v4 objective is teaching the residual head on MOSI Train.

The motivating observation is structural: `DISTILL / PRESERVE / ABSTAIN` only controls the
selective KD/preserve terms. The ordinary missing-modality supervised task loss remains active on
every missing event, including `ABSTAIN`. Because v8-v10 train only the residual head, the
bank-relevant objective is therefore:

```
SUPERVISED_MISSING(all missing events)
+ DISTILL_KD(on DISTILL events)
+ 0.25 * PRESERVE(on PRESERVE events)
```

`ABSTAIN` means **no KD and no preserve**, not “no residual update”.

## OOF protocol

- Seed: 1113.
- Reuse exactly the frozen v10 deterministic video-grouped 5-fold assignment.
- Each Train sample is audited only by the residual bank whose training excluded that sample's
  complete video.
- Use all three missing modes LA/LV/L for every held-out sample: 1284 × 3 = 3852 OOF events.
- No optimizer step is taken.
- No new checkpoint is selected.
- Official Test is never constructed or accessed.
- Official Valid is not used in audit statistics or decisions. The legacy v4 frozen-asset loader
  materializes its immutable Valid reference as an implementation prerequisite.

## Event audit

For every OOF Train event save:

- baseline, S0, current residual-student, Teacher predictions;
- residual delta and S0/current errors;
- `DISTILL / PRESERVE / ABSTAIN` branch;
- Teacher-beneficial and S0-beneficial flags;
- Teacher-safe and preserve-safe targets;
- whether the active selective target is locally safe;
- whether the current residual crossed the label or frozen baseline from S0;
- the fact that supervised missing regression remains active even on `ABSTAIN`.

The safe-projection design should make active DISTILL/PRESERVE targets locally non-worsening for
the event that generated them. If OOF beneficial samples are nevertheless harmed, the audit then
looks for shared-parameter influence rather than claiming the local target points in the wrong
direction.

## Gradient audit

At each frozen v10 conservative fold checkpoint, decompose the residual-head training gradient into:

- `SUPERVISED_BRANCH_DISTILL`
- `SUPERVISED_BRANCH_PRESERVE`
- `SUPERVISED_BRANCH_ABSTAIN`
- `SUPERVISED_TEACHER_BENEFICIAL`
- `SUPERVISED_TEACHER_NONBENEFICIAL`
- `DISTILL_KD`
- `PRESERVE_SCALED`
- derived `SUPERVISED_ALL`
- derived `SELECTIVE_ONLY`
- derived `TOTAL_RESIDUAL_OBJECTIVE`

For the OOF holdout, compute MAE gradients for:

- all events;
- Teacher-beneficial / Teacher-nonbeneficial;
- S0-beneficial / S0-nonbeneficial.

For each pair, report the dot product and cosine between the Train component gradient and OOF-loss
gradient.

With gradient descent update `theta <- theta - eta * g_train`, first-order OOF loss change is:

```
dL_oof ≈ -eta * dot(g_oof, g_train)
```

Therefore:

- positive dot: the Train component locally **improves** that OOF group;
- negative dot: the Train component locally **harms** that OOF group.

This is a local first-order diagnostic at the frozen checkpoint, not an exact Adam trajectory replay
and not by itself causal proof.

## Key diagnostic question

The most important rows are `Mode=ALL`, `OOFGroup=OOF_TEACHER_BENEFICIAL`, especially:

- `SUPERVISED_BRANCH_ABSTAIN`
- `SUPERVISED_TEACHER_NONBENEFICIAL`
- `DISTILL_KD`
- `PRESERVE_SCALED`
- `SELECTIVE_ONLY`
- `SUPERVISED_ALL`
- `TOTAL_RESIDUAL_OBJECTIVE`

If the safe branch targets are locally safe but one of these shared training gradients is negative
against OOF-beneficial MAE in most folds, that supports residual-level shared-update interference.
If `SUPERVISED_ALL` or its ABSTAIN/nonbeneficial subcomponents are the harmful force, the next
mechanism experiment should change the objective rather than the architecture.

## Outputs

Under:

`result/missing_baseline/cfcompat_objective_audit_v11/mosi/train_oof/seed1113_dev/`

- `objective_audit_v11_summary.json`
- `objective_audit_v11_train_oof_events.csv`
- `objective_audit_v11_event_summary.csv`
- `objective_audit_v11_gradient_influence_by_fold.csv`
- `objective_audit_v11_gradient_influence_summary.csv`
- `objective_audit_v11_headline_gradient_matrix.csv`
- `objective_audit_v11_fold_manifest.csv`
