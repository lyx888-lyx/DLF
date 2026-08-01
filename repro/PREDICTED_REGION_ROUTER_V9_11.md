# V9.11 Predicted-Region Expert Router

## Purpose

V9.11 tests whether coarse sentiment regions are more learnable than per-sample
expert advantage. It reuses the completed V9.9 strict semantic OOF expert pool and
the original frozen V9.3 Validation/Test experts. No shadow expert is retrained.

The five regions are fixed:

```text
strong_negative: y < -1.5
negative:       -1.5 <= y < -0.5
boundary:       -0.5 <= y <= 0.5
positive:        0.5 < y <= 1.5
strong_positive: y > 1.5
```

The fixed semantic region-to-action map is:

```text
strong_negative -> strong_negative expert
negative        -> Anchor
boundary        -> boundary expert
positive        -> positive expert
strong_positive -> strong_positive expert
```

## What is tested

The true-region route is an upper bound because it uses the label-defined region.
V9.11 measures whether a grouped OOF classifier can predict that region without
labels, and then routes using the fixed map.

It reports these separately:

```text
Anchor
Anchor-output threshold region route
Predicted-region route
True-region route upper bound
Per-sample oracle upper bound
```

## Input features

The classifier receives only fold-aligned, deployment-available fields:

- four function-space predictions;
- Anchor and five candidate outputs;
- candidate distances to the four fixed region boundaries;
- the complete fixed 21-dimensional semantic signature of each specialist;
- aggregate semantic disagreement statistics.

No fold-local unaligned latent vector and no label-derived feature is used.

## Model and losses

Each router predicts a continuous sentiment score, four cumulative ordinal logits,
five direct class logits, five blended region probabilities, and correctness
confidence. Training combines ordinal BCE, weighted five-class cross entropy, score
regression, expected-region regression, and confidence calibration. Early stopping
also includes the routed MAE obtained from the fixed semantic map.

## Grouped cross-fitting

For each of five grouped coach folds:

1. choose the epoch using inner-train and inner-validation groups;
2. refit a new classifier for that fixed epoch count on the full development set;
3. predict the untouched coach holdout groups;
4. save the refit model as one deployment ensemble member.

Validation and Test average the five fold models. Test labels never select an epoch,
mapping, threshold, or safety profile.

## Region-to-action maps

Two maps are pre-registered:

1. the fixed semantic map;
2. a Train-only empirical map estimated from per-region action MAE with shrinkage
   toward global action MAE.

During OOF, each empirical map is estimated only from that fold's development
groups and applied to its untouched holdout groups. The mapping family with lower
OOF routed MAE is fixed before Validation. The final empirical map, when selected,
is estimated from all Train samples only.

## Safety profiles

Three gates are pre-registered: `conservative`, `balanced`, and `broad`. They use
maximum region probability, top-two probability margin, entropy, model confidence,
blend strength, and a maximum activation rate. OOF eligibility requires positive
gain, nonnegative grouped-bootstrap lower bound, severe-harm rate at most 3%, and
stability across coach folds. Validation chooses only among OOF-eligible profiles;
otherwise deployment remains Anchor.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/predicted-region-router-v9-11
git pull origin innovation/predicted-region-router-v9-11

python3 scripts/smoke_test_predicted_region_router_v9_11.py

GPU=0 SEED=1111 NUM_WORKERS=1 \
  bash scripts/run_mosi_predicted_region_router_v9_11.sh \
  2>&1 | tee mosi_predicted_region_router_v911_seed1111.log
```

Required existing pool:

```text
result/semantic_cost_coach_v99/mosi/seed_1111/
  strict_semantic_expert_pool/strict_semantic_expert_pool_v99.pth
```

Output root:

```text
result/predicted_region_router_v911/mosi/seed_1111/
```

Important outputs:

```text
predicted_region_fold_0_v911.pth ... predicted_region_fold_4_v911.pth
v911_region_oof_predictions.csv
v911_region_oof_confusion.csv
v911_region_action_maps.csv
v911_region_ensemble.csv
v911_oof_policy_profiles.csv
v911_validation_profile_selection.csv
v911_test_summary.csv
v911_test_region_confusion.csv
predicted_region_router_v911_predictions.csv
predicted_region_router_v911_summary.json
```

The central question is whether strict OOF predicted-region routing materially
improves on Anchor while keeping harm controlled. The true-region score remains an
upper bound, not a deployable result.
