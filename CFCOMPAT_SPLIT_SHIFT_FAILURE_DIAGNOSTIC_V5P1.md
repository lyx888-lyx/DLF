# CFCompatKD v5.1 — Train→Valid Split-Shift / Failure Diagnostic

## Purpose

This branch does **not** propose a new distillation mechanism. It analyzes why the v5 Teacher-benefit relation is strong inside Train (grouped OOF) but fails to generalize to Valid.

The diagnostic is intentionally failure-centered:

1. identify which sample × missing-mode events the gate gets wrong;
2. distinguish false-positive routing (would distill when Teacher is not beneficial) from false-negative routing (misses genuinely beneficial Teacher cases);
3. measure Train→Valid feature distribution shift;
4. measure Train→Valid feature→Teacher-advantage relationship weakening or sign/orientation flips;
5. rank shared categorical patterns enriched among Valid failures.

## Protocol boundary

- Development seed remains **1113**.
- Inputs are the already frozen v5 gate-only CSV artifacts.
- The script constructs no DataLoader and trains no Student.
- It does not select checkpoints or thresholds.
- **Official Test is forbidden and is not constructed or read.**
- Results are diagnostic evidence only. They must not be used to retroactively relax or reinterpret the frozen v5 pre-screen.

This means the branch can answer "what fails from Train→Valid?" but cannot claim which specific MOSI Test clips failed. The historical one-time Test viability probe did not preserve sample-level Test artifacts.

## Frozen inputs

Default paths:

```text
result/missing_baseline/cfcompat_crossfit_transfer_risk_v5/mosi/valid_screen/seed1113_dev/gate_only/crossfit_transfer_risk_v5_train_oof_gate.csv
result/missing_baseline/cfcompat_crossfit_transfer_risk_v5/mosi/valid_screen/seed1113_dev/gate_only/crossfit_transfer_risk_v5_valid_gate_diagnostic.csv
```

The Train probability is `oof_benefit_probability`; the Valid probability is `full_train_benefit_probability`.

## Failure taxonomy

For each sample × missing mode:

- `true_positive`: Teacher is beneficial and gate predicts beneficial;
- `false_negative`: Teacher is beneficial but gate rejects it;
- `false_positive`: Teacher is not beneficial but gate predicts beneficial;
- `true_negative`: Teacher is not beneficial and gate rejects it.

Teacher condition is additionally split into:

- `strong_harmful`: advantage ≤ -0.02;
- `mild_harmful`: -0.02 < advantage < 0;
- `ambiguous_positive`: 0 ≤ advantage < 0.02;
- `beneficial`: advantage ≥ 0.02.

This explicitly separates **selection failure** (wrong Teacher-usefulness decision) from later Student optimization/interference questions.

## Outputs

The default output directory is:

```text
result/missing_baseline/cfcompat_split_shift_failure_diagnostic_v5p1/mosi/seed1113_train_valid
```

Files:

- `sample_failure_table.csv` — enriched per-event Train + Valid table;
- `feature_shift_summary.csv` — mean/std/quantiles, SMD, KS by ALL/LA/LV/L;
- `benefit_relationship_shift.csv` — Spearman and univariate AUC relation changes, including sign/orientation flips;
- `feature_benefit_relationship_bins.csv` — Train-quantile-binned feature→benefit curves evaluated on both splits;
- `mode_failure_summary.csv` — gate error / FP / FN rates by split and missing mode;
- `failure_interaction_rules.csv` — one- and two-factor failure patterns with Valid-vs-Train enrichment;
- `top_gate_failure_samples.csv` — highest-confidence / highest-Brier gate mistakes, preserving sample IDs;
- `split_shift_failure_summary.json` — compact ranked diagnostic summary and protocol guardrails.

## Run on Windows

```powershell
.\scripts\run_windows_cfcompat_split_shift_failure_diagnostic_v5p1.ps1 -Overwrite
```

The wrapper first compiles the diagnostic, runs a synthetic smoke test, verifies both frozen input CSVs exist, and then produces the diagnostic outputs.

## How to interpret the results

Prioritize three patterns:

1. **Distribution shift**: large absolute SMD / KS means the raw geometry itself moved.
2. **Relationship shift**: Train correlation/AUC strong but Valid weak or reversed means the same geometry no longer implies the same Teacher usefulness.
3. **Failure concentration**: high Valid misclassification lift for rules such as `mode=L & baseline_difficulty_quartile=Q1_low` indicates a concrete subgroup to inspect sample-by-sample.

A large distribution shift alone does not prove the mechanism. The strongest evidence is when a feature is predictive of Teacher advantage in Train, loses or flips that relation in Valid, and the corresponding subgroup is enriched for high-confidence Valid gate mistakes.

## Next decision

Do **not** immediately replace logistic regression with a stronger classifier. Use this diagnostic to decide which mechanism class is justified:

- calibration/scale shift → rank, percentile, or mode-normalized invariant features;
- Teacher-usefulness relation instability → perturbation-consistency / shift-stability selection;
- failures remain common even when Teacher is genuinely beneficial → investigate shared-parameter gradient interference rather than gating alone.
