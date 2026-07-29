# MOSI V9: Region-Balanced Ordinal Soft Mixture

## Goal

Test whether the unusually high error in the ordinary-positive region can be reduced without using Test labels, Test statistics, or post-hoc Test calibration.

V9 starts from the frozen V7.1 complementarity Student and adds only a guarded prediction head:

1. Train-frequency-balanced Huber regression over five sentiment regions.
2. A soft negative/neutral/positive gate.
3. Three polarity-constrained regression experts.
4. Monotonic Acc-7 and Acc-5 ordinal auxiliary heads.
5. A cross-zero penalty for confidently non-neutral labels.
6. A bounded blend back to the frozen V7.1 Student.

The first run keeps GroupDRO disabled. It is an isolated later ablation, not part of the default result.

## Five-region definition

With `strong_threshold=1.0` and `neutral_radius=1e-6`:

- strong negative: `y < -1`
- ordinary negative: `-1 <= y < 0`
- neutral: `y == 0` up to the configured tolerance
- ordinary positive: `0 < y <= 1`
- strong positive: `y > 1`

The exact counts and normalized weights are written to `v9_train_region_weights.csv`.

## Leakage discipline

- Region weights: Train labels only.
- Model training: Train only.
- Epoch, committee, beta, and alpha: Validation only.
- Test: collected once after the complete policy is frozen.
- The original V7.1 hybrid is always included as a legal fallback.

## First run

```bash
bash scripts/run_mosi_region_balanced_ordinal_mixture_v9.sh \
  2>&1 | tee mosi_region_balanced_ordinal_mixture_v9.log
```

Output directory:

```text
result/region_balanced_ordinal_mixture_v9/mosi/seed_1111/
```

## Files to inspect first

```text
v9_train_region_weights.csv
v9_training_history.csv
v9_valid_policy_search.csv
v9_test_comparison.csv
v9_test_region_diagnostics.csv
v9_test_predictions.csv
region_balanced_ordinal_mixture_v9_summary.json
```

## Acceptance order

1. The frozen V7.1 reference is reproduced exactly.
2. Validation selects V9 without exceeding the configured MAE/fold constraints.
3. Ordinary-positive signed bias and cross-zero rate move in the intended direction.
4. Overall Test MAE does not degrade materially.
5. Bootstrap ordinary-positive gain is reported but not used for selection.

Run only Seed 1111 first. Do not start five seeds until ID alignment, reference reproduction, and the single-seed diagnostics are verified.
