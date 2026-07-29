# MOSI V9.2: Constrained Positive-Residual Specialist

## Motivation

V9.1 showed a real positive-region learning signal:

- the standalone mixture reduced ordinary-positive Validation MAE substantially;
- the same generalist mixture harmed other regions;
- the final Valid selector therefore chose zero shrinkage and did not use training.

V9.2 converts that signal into a local, auditable correction instead of asking a
new head to solve the full `[-3, 3]` regression task.

## Model

The historical V7.1 backbone and residual Student are restored and frozen.

A new specialist receives the stop-gradient fusion feature and predicts:

```text
gate_probability = sigmoid(gate_head(h))
magnitude        = max_correction * sigmoid(magnitude_head(h))
positive_delta   = gate_probability * magnitude
```

Therefore:

```text
0 <= positive_delta <= max_correction
```

The specialist cannot directly lower any prediction.

## Training target

For each Train sample, reconstruct the frozen V7.1 hybrid reference using the
aligned committee cache and the historical V7.1 policy.

```text
raw_residual = label - frozen_reference
target_delta = clamp(raw_residual, 0, max_correction)
```

`target_delta` is forced to zero for non-positive labels. The gate target is one
only when `target_delta > residual_margin`.

Training uses:

1. region-weighted Huber on `positive_delta`;
2. gate BCE;
3. magnitude regression on active targets;
4. no-harm penalty when target correction is zero;
5. over-correction penalty;
6. sparse-gate regularization.

Ordinary-positive Train samples receive a modest additional boost. Their
Validation MAE is diagnostic only and is not a policy-selection term.

## Staged selection

### Stage A: specialist checkpoint

The specialist checkpoint is selected on Validation with the fixed candidate:

```text
frozen_v71_reference + positive_delta
```

The objective is driven by overall MAE with small worst-region, fold-stability,
and non-positive MAE terms.

### Stage B: attribution policy

After the specialist checkpoint is frozen, run one compact Validation search:

```text
legacy_reference
legacy_beta_search
zero_shrinkage
positive_residual
```

For a positive-residual candidate:

```text
base = beta * committee
     + (1 - beta) * (1 - shrinkage) * legacy_student

prediction = base + gamma * positive_delta
```

The policy must satisfy overall MAE, fold MAE, worst-region, and non-positive
MAE constraints. Test is collected once after both stages are frozen.

## First run

```bash
bash scripts/run_mosi_positive_residual_specialist_v92.sh \
  2>&1 | tee mosi_positive_residual_specialist_v92.log
```

Output:

```text
result/positive_residual_specialist_v92/mosi/seed_1111/
```

Audit:

```bash
python3 scripts/audit_positive_residual_specialist_v92.py \
  --run-dir result/positive_residual_specialist_v92/mosi/seed_1111 \
  --expected-reference-mae 0.6994764 \
  --reference-tolerance 0.0002
```

## Interpretation

Training contributes to the final result only when:

```text
selected_policy.source == positive_residual
selected_policy.gamma > 0
training_contributed == true
```

Other outcomes mean:

- `legacy_reference`: no Valid-supported change;
- `legacy_beta_search`: committee reweighting only;
- `zero_shrinkage`: output shrinkage only;
- `positive_residual`: the trained local specialist is used.

## Files to inspect

```text
v92_training_history.csv
v92_valid_policy_search.csv
v92_final_valid_policy.csv
v92_test_comparison.csv
v92_test_region_diagnostics.csv
v92_test_predictions.csv
positive_residual_specialist_v92_summary.json
```

Run Seed 1111 only. Do not start five seeds until reference reproduction, ID
alignment, correction bounds, and the attribution family are audited.
