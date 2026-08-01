# V9.10 Semantic Relative-Regret Coach

## Purpose

V9.10 reuses the completed V9.9 strict semantic OOF expert pool and the original
frozen V9.3 Validation/Test experts. It changes only the coaching target and
inference protocol.

V9.9 predicted five absolute action costs. The per-action cost prediction error was
far larger than the small difference between experts, and its conservative gain
subtracted two large marginal cost scales. V9.10 predicts each specialist's signed
error difference relative to Anchor instead.

## Relative-regret target

For specialist `k`:

```text
delta[k] = abs(prediction[k] - label) - abs(anchor - label)
```

Interpretation:

```text
delta[k] < 0  specialist k beats Anchor
delta[k] > 0  specialist k is worse than Anchor
```

The model predicts for each of the four specialists:

```text
predicted_delta[k]
predicted_scale[k]
P(delta[k] < 0)
```

Anchor is an explicit action with fixed regret zero. The unconstrained selector
minimizes:

```text
predicted_delta[k] + risk_aversion * predicted_scale[k]
```

across Anchor and four specialists. If every specialist has positive adjusted
regret, Anchor is selected automatically.

The safety lower gain for the actually selected specialist is:

```text
lower_gain = -(predicted_delta + z * predicted_scale)
```

This uses uncertainty of the relative difference directly. It does not add two
large absolute-cost uncertainties.

## Training losses

Training uses:

- weighted Smooth-L1 relative-regret regression;
- Laplace-style relative-regret scale calibration;
- Beat-Anchor binary classification;
- signed-margin consistency against Anchor;
- pairwise ranking across Anchor and four specialists;
- oracle-action cross entropy;
- differentiable expected selected regret.

## Fold-refit ensemble

For each of five grouped coach folds:

1. choose the training epoch using inner-train and inner-validation groups;
2. initialize a new coach;
3. refit it for that fixed epoch count on the complete fold development set;
4. predict the untouched coach holdout groups;
5. save the refit model as one deployment ensemble member.

Validation and Test average the five fold models. Total uncertainty combines mean
predicted relative scale and between-model disagreement. No median-epoch full model
is used.

## Expert pool

V9.10 does not retrain the shadow experts. It expects the strict V9.9 semantic pool:

```text
result/semantic_cost_coach_v99/mosi/seed_1111/
  strict_semantic_expert_pool/strict_semantic_expert_pool_v99.pth
```

That pool remains fully nested and holdout-label isolated. Validation/Test continue
to use the original frozen V9.3 experts with the same semantic signature schema.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/relative-regret-coach-v9-10
git pull origin innovation/relative-regret-coach-v9-10

python3 scripts/smoke_test_relative_regret_coach_v9_10.py

GPU=0 SEED=1111 NUM_WORKERS=1 \
  bash scripts/run_mosi_relative_regret_coach_v9_10.sh \
  2>&1 | tee mosi_relative_regret_coach_v910_seed1111.log
```

## Output root

```text
result/relative_regret_coach_v910/mosi/seed_1111/
```

Important outputs:

```text
relative_regret_fold_0_v910.pth ... relative_regret_fold_4_v910.pth
v910_relative_regret_oof_predictions.csv
v910_relative_regret_ensemble.csv
v910_oof_policy_profiles.csv
v910_validation_profile_selection.csv
v910_test_summary.csv
relative_regret_coach_v910_predictions.csv
relative_regret_coach_v910_summary.json
```

## Primary diagnostics

The main OOF diagnostics are:

```text
oof_per_expert_delta_mae
oof_per_expert_delta_spearman
oof_per_expert_beat_accuracy
oof_selected_action_accuracy
oof_regret_selected_mae
oof_unrestricted_gain
oof_selected_benefit_rate
oof_selected_harm_over_010_rate
```

The main Test diagnostics are:

```text
minimum_predicted_regret_action_all
risk_adjusted_regret_action_all
relative_regret_soft_mixture_all
relative_regret_valid_selected
```

Only `relative_regret_valid_selected` is the deployable result. The unrestricted
routes are diagnostics and never tune thresholds on Test labels.
