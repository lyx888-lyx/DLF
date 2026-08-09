# CFCompatKD v12.2 — Exact-Replay Trajectory Train-OOF Audit

## Purpose

v12.2 is diagnostic only. It does **not** define or select a new candidate model.
It tests the mechanism suggested by v12.1: the frozen v12 conservative endpoint
has a beneficial-safe local `SURGERY_UPDATE`, yet the final function is still
less safe than v8. The remaining question is whether harmful directions existed
earlier in the v12 trajectory and later reversed, or whether full-Train
first-order gradient alignment is insufficient to explain the finite Adam path.

## Exact replay requirement

The formal numerical-hotfix v12 training function is reused directly. The only
training-time instrumentation is a read-only hook around the existing
`bank_state_cpu(student)` epoch snapshot call. No extra OOF forward/backward
pass is allowed until **all five folds finish training**.

Before any trajectory result is interpreted, every fold must reproduce:

- the frozen v12 conservative selected epoch exactly;
- the frozen v12 absolute-best Train-video-holdout epoch exactly;
- the frozen v12 missing-mode sequence SHA256 exactly;
- the frozen v12 conservative residual-bank tensor state exactly.

A mismatch stops the audit. It must not be explained as trajectory evidence.

## Post-hoc trajectory observations

After 5/5 replay verification, every captured epoch state is evaluated on that
fold's held-out Train videos using exhaustive LA/LV/L views. Per-event records
include baseline, Frozen-S0, residual student, Teacher, residual delta,
DISTILL/PRESERVE/ABSTAIN branch, Teacher-beneficial and S0-beneficial flags,
negative-transfer/severe-negative-transfer flags, label/baseline crossing, and
regression relative to S0.

The aggregate trajectory is reported for:

- all missing events;
- Teacher-beneficial;
- Teacher-nonbeneficial;
- S0-beneficial;
- S0-nonbeneficial;
- S0-and-Teacher-beneficial.

Failure-onset and synchronized three-mode clip summaries are also saved.

## Frozen gradient milestones

Before the run, the fixed milestone epochs are:

`0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64`

A milestone is audited only if that fold reached it. The conservative-selected
and absolute-best epochs are also audited because they come from the already
frozen Train-video selector, not from v12.2 OOF gradient results.

At each audited state, v12.2 computes:

- `SUPERVISED_ALL_RAW`;
- `SELECTIVE_ONLY = DISTILL_KD + 0.25 * PRESERVE`;
- `SUPERVISED_PROJECTED`, from the analytic v12 asymmetric projection;
- `SUPERVISED_REMOVED_CONFLICT`;
- `SURGERY_UPDATE = SUPERVISED_PROJECTED + SELECTIVE_ONLY`.

These directions are compared by dot product/cosine with OOF MAE gradients for
Teacher-beneficial, Teacher-nonbeneficial, S0-beneficial, and S0-nonbeneficial
groups. Positive dot means a gradient-descent step on the Train direction is
predicted to decrease that OOF MAE locally; negative dot predicts local harm.

## Interpretation frozen before results

1. If early fixed milestones show `SURGERY_UPDATE` harming beneficial OOF groups
   and the conservative endpoint later improves them, this supports a
   trajectory sign-reversal / accumulated functional-drift mechanism.
2. If `SURGERY_UPDATE` is already beneficial-safe at all early milestones while
   actual beneficial OOF failures still accumulate, expected full-Train
   first-order geometry is insufficient. The next audit should move to the
   finite optimizer-update level: Adam momentum/second moments, update-window
   heterogeneity, and microbatch-specific interference.
3. If signs differ strongly by fold, treat the mechanism as fold-conditional
   rather than a single common trajectory.

v12.2 has no model-promotion gate and does not tune any threshold from the
result.

## Split protocol

- Train is used for the exact v12 replay and video-grouped OOF diagnostics.
- Official Valid is not used in trajectory statistics, gradient milestones, or
  any v12.2 decision. The legacy v4 asset loader may materialize its immutable
  Valid reference as a prerequisite.
- Official Test is never constructed or accessed.

## Run

```powershell
git fetch origin refs/heads/agent/cfcompat-trajectory-audit-v12p2
git switch -C agent/cfcompat-trajectory-audit-v12p2 FETCH_HEAD
.\scripts\run_windows_cfcompat_trajectory_audit_v12p2.ps1 -Overwrite
```

The runner stops immediately if compile, smoke, exact replay, or the formal
audit fails.
