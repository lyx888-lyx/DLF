# Regret-Aware Preserve-or-Distill CFCompatKD v4

## Research question

CFCompatKD can improve some missing-modality examples while moving other
examples away from a frozen missing-modality baseline that was already closer
to the label.  v4 targets that sample-level negative-transfer failure directly.

This is an exploratory **MOSI Seed-1113 development screen**.  It trains exactly
one new trajectory.  Existing audited v2 `cfcompat_replay` and
`student_safe_uniform` artifacts are reused as frozen Valid references and are
not retrained.

Official Test remains forbidden.

## Frozen anchor

For each Train sample and missing mode (LA/LV/L), v4 caches the prediction of
the validation-best frozen ModDrop evaluator:

```text
b = frozen ModDrop missing-modality prediction
s = current missing-modality Student prediction (detached for decisions)
t = frozen full-modality Teacher prediction
y = Train label
c = frozen CFCompat compatibility
```

The ModDrop prediction is the same baseline family used by the existing
counterfactual-compatibility and Valid negative-transfer analysis.  The cache is
Train-only and contains 1284 samples x three possible missing modes.

## Three-way decision

### 1. DISTILL

Teacher must beat the frozen baseline by the fixed margin 0.02:

```text
|b-y| - |t-y| >= 0.02
```

The Teacher is then clipped to the closed interval between the current Student
and label:

```text
t_safe = clip(t, min(s,y), max(s,y))
```

DISTILL is active only if `t_safe != s`.

CFCompat is retained only as a mild prior:

```text
w_distill = 0.75 + 0.25*c
```

The distillation term is a gated SmoothL1 loss toward `t_safe`.

### 2. PRESERVE

If DISTILL is unavailable and the current Student has regressed from the frozen
baseline by at least 0.02:

```text
|s-y| - |b-y| >= 0.02
```

v4 activates a preservation loss.  The baseline itself is projected onto the
current-Student-to-label interval:

```text
b_safe = clip(b, min(s,y), max(s,y))
```

This prevents the preservation anchor from crossing the label and becoming a
new overshoot.  The preservation coefficient is frozen at:

```text
lambda_preserve = 0.25
```

### 3. ABSTAIN

If neither DISTILL nor PRESERVE is selected, the auxiliary KD/preservation
signal is zero and the sample is trained only by the ordinary DLF supervised
objectives and shared model updates.

The three branches are mutually exclusive and exhaustive.

## Training objective

```text
L = L_full_supervised
  + L_missing_supervised
  + L_distill
  + 0.25 * L_preserve
```

No utility/difficulty weighting from v3 is used.

## Why this differs from v2

v2 only stopped an unsafe sample's own KD gradient.  It did not protect an
already-good frozen baseline prediction from being damaged by updates caused by
other samples and shared parameters.

v4 adds an explicit preservation force only when the current Student has
measurably regressed relative to the frozen ModDrop anchor.

## New Valid metric: negative-transfer rate

For each Valid event:

```text
regret = candidate_error - frozen_baseline_error
```

A negative-transfer event is counted when:

```text
regret > 0.02
```

The principal v4 safety statistic pools only LA/LV/L (`Mode=MISSING_ALL`).
A severe-negative-transfer diagnostic also uses margin 0.10.

## Frozen Seed-1113 promotion checks

The single-seed candidate is promoted to a new frozen three-seed Valid screen
only if every check passes:

- Valid-J is no more than 0.002 worse than v2 `student_safe_uniform`.
- Valid-J is no more than 0.005 worse than original CFCompat replay.
- Missing-modality negative-transfer rate is reduced by at least 0.02 absolute
  versus v2 Uniform.
- Harmful imitation does not increase versus v2 Uniform.
- Q1-easy gain is no worse than v2 Uniform by more than 0.002.
- Q4-hard gain is no worse than v2 Uniform by more than 0.002.
- `better_and_correct` gain is no worse than v2 Uniform by more than 0.002.
- At least two epochs are within the 0.002 Valid-J noninferiority band versus
  v2 Uniform.
- DISTILL, PRESERVE and ABSTAIN are all exercised.

A pass yields:

```text
PROMOTE_REGRET_PRESERVE_TO_3SEED_VALID_SCREEN
```

A failure yields:

```text
STOP_REGRET_PRESERVE_SINGLE_SEED_DEV_FAILED
```

Neither verdict authorizes official Test.

## Windows commands

Switch to the implementation branch and verify the exact HEAD shown in the
conversation before running.

Two-epoch real-chain smoke:

```powershell
.\scripts\run_windows_cfcompat_regret_preserve_valid_screen.ps1 -Overwrite
```

Full Seed-1113 development trajectory:

```powershell
.\scripts\run_windows_cfcompat_regret_preserve_valid_screen.ps1 -Formal -Overwrite
```

Audit only, without retraining:

```powershell
.\scripts\audit_windows_cfcompat_regret_preserve_valid_screen.ps1
```

## Main outputs

```text
result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev
```

Important files include:

- `regret_preserve_v4_candidate_grid.csv`
- `regret_preserve_v4_all_epoch_metrics.csv`
- `regret_preserve_v4_negative_transfer_metrics.csv`
- `regret_preserve_v4_train_baseline_cache.csv`
- `regret_preserve_v4_train_decisions.csv`
- `regret_preserve_v4_group_metrics.csv`
- `regret_preserve_v4_valid_screen_summary.json`
- `regret_preserve_v4_valid_screen_report.md`
- `regret_preserve_v4_source_manifest.json`
- `regret_preserve_v4_audit_check.json` after the independent audit

The Train decision artifact records every event's baseline/Student/Teacher
prediction, decision branch, safe targets, compatibility, branch gates and raw
per-event losses so the three-way policy can be recomputed independently.
