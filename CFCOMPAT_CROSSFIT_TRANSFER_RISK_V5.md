# Cross-Fitted Transfer-Risk CFCompatKD v5

## Motivation

The frozen Seed1113 v4 Regret-Preserve candidate improved MOSI Valid but the one-time MOSI Test viability probe did not show a positive cross-split signal.  v5 therefore stops adding sample-local preservation heuristics and targets the generalization problem directly: predict whether frozen Teacher knowledge is beneficial for an unseen sample using label-free descriptors.

## Development scope

- Development seed: **1113 only**.
- New Student trajectories: **at most one** in the formal run.
- Frozen references: audited v2 `cfcompat_replay` / `student_safe_uniform` and frozen audited v4 candidate.
- Official Test: **forbidden** in this branch.  The prior Test viability probe is not re-opened.

## Benefit target

For each Train sample and missing mode:

- `b`: frozen validation-best ModDrop missing prediction.
- `t`: frozen full-modality Teacher prediction.
- `y`: Train label.

The gate supervision target is

`beneficial = 1[ |b-y| - |t-y| >= 0.02 ]`.

The label is used only to create this supervision target.  It is never included in the gate feature vector.

## Label-free gate features

The fixed feature vector contains frozen predictions and their signed/absolute discrepancies:

- frozen ModDrop missing prediction;
- frozen ModDrop full prediction;
- frozen Teacher full prediction;
- initial missing-modality Student prediction;
- signed and absolute Teacher/baseline/initial-Student gaps;
- frozen baseline full-vs-missing discrepancy;
- LA/LV/L one-hot indicators.

Sample IDs and labels are not gate features.

## Grouped cross-fitting

- 5 folds, seed `20260807`.
- Split unit is **sample_index**, not event row.
- A sample's LA/LV/L events must remain in the same fold.
- The Student trajectory is routed only by `oof_benefit_probability` produced by a model that never trained on that sample.
- A full-Train logistic model is fitted only to measure gate generalization on Valid; its Valid probabilities never route Train events.

The classifier is `StandardScaler + LogisticRegression(C=1.0, lbfgs)`.

## Gate pre-screen

Before a full Student trajectory is allowed:

- Train OOF ROC-AUC >= `0.58`;
- Valid ROC-AUC >= `0.55`;
- Valid Brier score must not exceed the constant Train-prevalence predictor;
- all grouped folds must contain both classes.

Use `-GateOnly` first.  A failed formal pre-screen stops before Student training.

## Student objective

The Teacher target retains the current-Student-safe projection:

`t_safe = clip(t, min(s,y), max(s,y))`.

Let `p` be the grouped OOF benefit probability.  The risk weight is

`w = max(2p - 1, 0)`.

The effective gate is `active * w`, where `active` means `t_safe != s`.

Unlike earlier normalized gated KD, v5 preserves attenuation:

`L_KD = sum(active * w * SmoothL1(student, t_safe)) / (sum(active) + eps)`.

Therefore low confidence genuinely lowers KD magnitude instead of merely redistributing a fixed KD budget.

Total loss:

`L = L_full + L_missing + L_KD`.

v5 intentionally has no Train-label-triggered Preserve branch and no best-so-far memory.

## Frozen Seed1113 candidate gate

After the gate pre-screen passes, the one candidate trajectory must satisfy all of:

- Valid J no worse than frozen original CFCompat replay by more than `0.002`;
- Valid J no worse than v4 by more than `0.005`;
- MISSING_ALL negative-transfer rate improves by at least `0.02` absolute vs replay;
- severe negative-transfer rate no worse than replay by more than `0.01`;
- positive-transfer rate no worse than replay by more than `0.01`;
- harmful-imitation rate no worse than replay by more than `0.01`;
- Q1/Q4 gain no worse than replay by more than `0.005`;
- better-and-correct gain retains v4 within `0.005`;
- at least two epochs are within replay J + `0.002`;
- risk weighting is exercised and non-collapsed.

Pass verdict:

`PROMOTE_CROSSFIT_TRANSFER_RISK_V5_TO_3SEED_VALID_SCREEN`

Fail verdict:

`STOP_CROSSFIT_TRANSFER_RISK_V5_SINGLE_SEED_DEV_FAILED`

## Windows commands

Cheap gate-only screen:

```powershell
.\scripts\run_windows_cfcompat_crossfit_transfer_risk_valid_screen.ps1 -GateOnly -Overwrite
```

Two-epoch real-chain smoke:

```powershell
.\scripts\run_windows_cfcompat_crossfit_transfer_risk_valid_screen.ps1 -Overwrite
```

Formal Seed1113 run, only after gate-only passes:

```powershell
.\scripts\run_windows_cfcompat_crossfit_transfer_risk_valid_screen.ps1 -Formal -Overwrite
```

Audit only:

```powershell
.\scripts\audit_windows_cfcompat_crossfit_transfer_risk_valid_screen.ps1
```
