# V9.7 Attainable Frontier Coach

## Goal

V9.7 implements the intended mechanism directly:

1. produce five candidate predictions for each sample;
2. identify the candidate value that actually achieves the smallest absolute
   error in Train OOF;
3. train a coach to predict that *attainable frontier value* and its gain over
   Anchor;
4. at deployment, choose the candidate prediction nearest the predicted
   frontier value;
5. retain Anchor unless a pre-registered safety profile permits the call.

The candidate schema is:

```text
anchor
strong_negative
boundary
positive
strong_positive
```

Unlike V9.6, the coach input explicitly contains all five candidate values, all
four expert corrections, all four expert confidences, candidate order statistics,
and the aligned four-dimensional function-space prediction vector.

## Train OOF candidate pool

V9.7 uses the original V9.3 specialist architectures:

- Boundary and Positive use `RoleConditionedDLF` with the V9 loss and staged
  head/tail schedule.
- Strong Negative and Strong Positive use `CachedTailResidualHeadV92` with the
  V9.2 OOF tail loss.

The V9.2 grouped OOF fold assignment is reused. For every outer fold, the
complete holdout groups are excluded from optimization of all four fold-local
specialists. The holdout predictions are then assembled into one 1284-sample
candidate pool.

The existing V9 teacher checkpoints, teacher cache, and validation-fitted teacher
weights are frozen artifacts. They are not retrained inside every OOF fold. The
protocol therefore guarantees holdout-label isolation for the fold-local
specialists, but it is not a fully nested retraining of the entire historical
teacher stack. This limitation is recorded in the pool provenance and printed by
the audit.

## Frontier targets

For sample `i` and candidate `k`:

```text
cost[i,k] = abs(candidate[i,k] - label[i])
oracle_action[i] = argmin_k cost[i,k]
oracle_value[i] = candidate[i, oracle_action[i]]
oracle_gain[i] = cost[i,anchor] - min_k cost[i,k]
oracle_cost_margin[i] = second_smallest_cost - smallest_cost
```

The coach predicts:

```text
attainable frontier value
oracle gain over Anchor
probability that gain exceeds 0.02
auxiliary oracle action logits
```

The predicted frontier value is constrained to the interval between the minimum
and maximum candidate predictions. The nearest candidate to that value is the
proposed action.

## Coach cross-fitting

The frontier coach is source-video grouped cross-fitted on the Train OOF
candidate pool. Every OOF coach prediction excludes its complete source-video
group. The selected fold epochs are summarized by their median, and a final coach
is refit on the complete Train OOF pool.

## Pre-registered routing

Only three policies exist:

```text
conservative
balanced
broad
```

They have fixed thresholds for predicted gain, useful-call probability, nearest
candidate margin, expert confidence, blend beta, and maximum activation rate.
They are first screened on grouped Train OOF predictions using:

- minimum OOF MAE gain;
- source-video bootstrap lower gain bound;
- maximum harm rate;
- maximum activation rate;
- positive gain in most outer specialist folds.

Validation only compares Anchor with the OOF-eligible pre-registered profiles.
It does not search a large threshold grid. A non-Anchor profile must still exceed
the minimum Validation gain and satisfy the harm constraint. Otherwise the final
prediction is exactly Anchor.

## Requirements

V9.7 requires existing V9 and V9.2 artifacts:

```text
result/role_conditioned_experts_v9/mosi/seed_1111/
result/oof_tail_residual_experts_v92/mosi/seed_1111/
```

It does not require V9.4, V9.5, or V9.6 output.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/attainable-frontier-coach-v9-7
git pull origin innovation/attainable-frontier-coach-v9-7

python3 scripts/smoke_test_attainable_frontier_coach_v9_7.py

GPU=0 SEED=1111 \
  bash scripts/run_mosi_attainable_frontier_coach_v9_7.sh \
  2>&1 | tee mosi_attainable_frontier_coach_v97_seed1111.log
```

The role experts are full DLF models and are trained once per OOF fold, so the
initial frontier-pool build is substantially more expensive than V9.5/V9.6.
Fold caches support resume.

## Outputs

```text
result/attainable_frontier_coach_v97/mosi/seed_1111/
```

Important files:

```text
frontier_expert_pool/crossfit_v93_frontier_pool_v97.pth
frontier_expert_pool/v97_crossfit_frontier_pool_summary.json
frontier_expert_pool/v97_crossfit_frontier_pool.csv
v97_frontier_coach_oof_predictions.csv
v97_oof_policy_profiles.csv
v97_validation_profile_selection.csv
v97_test_summary.csv
attainable_frontier_coach_v97_predictions.csv
attainable_frontier_coach_v97_summary.json
```
