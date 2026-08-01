# V9.6 Distributional Target-Value Nearest-Expert Coach

## Goal

V9.6 keeps the strongest frozen expert pool from V9.3. It does not retrain any
expert and does not require V9.4 output.

Instead of predicting a five-region class or a five-action label, the coach
predicts the conditional target-value distribution for each sample. It emits
monotone `q10, q25, q50, q75, q90` values around the Anchor prediction.

The candidate actions are:

```text
anchor
strong_negative
boundary
positive
strong_positive
```

Two interpretable action-risk estimates are evaluated:

```text
median_distance = abs(action_prediction - q50)
quantile_risk   = integral_0^1 abs(action_prediction - Q(tau)) d tau
```

The integral is approximated by midpoint quadrature over the five predicted
quantiles. It estimates conditional expected absolute error and therefore aligns
with the final MAE objective.

The coach proposes the action with minimum estimated risk. A specialist is
actually used only when it improves estimated risk and median distance by a
minimum margin, has sufficient frozen-expert confidence, points toward the
predicted target, and the predicted 80% interval is sufficiently narrow.
Otherwise deployment remains at Anchor.

## Training protocol

The target-distribution model is trained by source-video grouped cross-fitting
on the V9.2 Train OOF cache. Every Train OOF target prediction is produced by a
model that excludes the complete source video group. A final target model is
then fit on the complete Train OOF cache for the median selected epoch.

The frozen V9.3 experts are collected only for Validation and Test. Validation
selects risk mode and routing safety thresholds. Test labels are used only for
final metrics and explicitly named diagnostic policies.

## Safety gate

A non-Anchor policy is eligible only when, by default:

- Validation MAE gain is at least `0.0015`;
- the source-video bootstrap 20th-percentile gain is non-negative;
- activation is positive and no greater than `35%`;
- the fraction harmed by more than `0.10` is no greater than `3%`.

If no policy passes, deployment is exactly Anchor.

## Diagnostics

The output compares:

```text
Anchor
Target median used directly
Nearest expert to target median without safety gate
Minimum quantile-risk expert without safety gate
Validation-selected safe route
True-region expert policy
True-label nearest-expert sample oracle
```

If target prediction were perfect, true-label nearest expert equals the V9.3
sample oracle, whose prior Seed 1111 Test MAE was `0.5926`.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/distributional-target-nearest-expert-v9-6
git pull origin innovation/distributional-target-nearest-expert-v9-6

python3 scripts/smoke_test_distributional_target_nearest_expert_v9_6.py

GPU=0 SEED=1111 \
  bash scripts/run_mosi_distributional_target_nearest_expert_v9_6.sh \
  2>&1 | tee mosi_distributional_target_nearest_expert_v96_seed1111.log
```

## Output root

```text
result/distributional_target_nearest_expert_v96/mosi/seed_1111/
```

Important files:

```text
v96_target_oof_predictions.csv
v96_target_fold_history.csv
distributional_target_coach_v96.pth
v96_route_calibration.csv
v96_test_summary.csv
distributional_target_nearest_expert_v96_predictions.csv
distributional_target_nearest_expert_v96_summary.json
```
