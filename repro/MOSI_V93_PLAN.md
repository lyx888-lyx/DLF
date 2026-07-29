# MOSI V9.3: Gateless Positive Residual

## Motivation

V9.1 showed that the frozen DLF fusion feature contains a useful ordinary-positive signal, but its full-range expert harmed other regions. V9.2 constrained the task to a bounded positive residual, but its learned gate collapsed:

```text
gate_mean ≈ 0.031
correction_mean ≈ 0.00087
correction_max ≈ 0.00219
```

V9.3 therefore removes the gate and tests the residual magnitude independently.

## Model

The complete V7.1 path is frozen. The new trainable path predicts only:

```text
0 <= delta(x) <= max_correction
```

The deployed candidate is selected from:

```text
legacy_reference
legacy_beta_search
zero_shrinkage
positive_residual
```

For a positive-residual candidate:

```text
prediction = selected_base + gamma * delta(x)
```

`gamma=0` remains available through the reference/beta/shrinkage families.

## Training target

Using the frozen V7.1 Hybrid reference:

```text
raw_residual = label - reference
target = clamp(raw_residual, 0, max_correction)
target = target only for positive-label samples
```

The direct loss contains:

- weighted bounded-residual Huber;
- an additional active-target Huber term;
- weak no-harm penalty on zero-target samples;
- weak over-correction penalty.

There is no gate BCE and no sparsity loss.

## Staged protocol

1. Train the residual magnitude on Train.
2. Select its checkpoint on Validation using residual prediction quality, before any final policy search.
3. Freeze the magnitude checkpoint.
4. Run one compact Validation-only search over committee, beta, shrinkage, and gamma.
5. Evaluate Test once.

Ordinary-positive Validation MAE is diagnostic only and is not directly optimized by policy selection.

## First run

```bash
bash scripts/run_mosi_gateless_positive_residual_v93.sh \
  2>&1 | tee mosi_gateless_positive_residual_v93.log
```

Audit:

```bash
python3 scripts/audit_gateless_positive_residual_v93.py \
  --run-dir result/gateless_positive_residual_v93/mosi/seed_1111 \
  --expected-reference-mae 0.6994764 \
  --reference-tolerance 0.0002
```

## Interpretation

A real training contribution requires:

```text
selected_policy.source == positive_residual
selected_policy.gamma > 0
training_contributed == true
```

Other outcomes mean:

- `legacy_reference`: no supported change;
- `legacy_beta_search`: committee reweighting only;
- `zero_shrinkage`: output shrinkage only.

Even when the final policy rejects the residual, the magnitude learning curve is informative. Success at the representation level requires a clear decrease in Validation residual Huber without correction collapsing near zero.

## Primary files

```text
v93_training_history.csv
v93_valid_policy_search.csv
v93_final_valid_policy.csv
v93_test_comparison.csv
v93_test_region_diagnostics.csv
v93_test_predictions.csv
gateless_positive_residual_v93_summary.json
```

Run Seed 1111 only until the frozen V7.1 reference, ID alignment, correction range, and attribution audit all pass.
