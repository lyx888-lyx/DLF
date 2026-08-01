# V9.9 Semantic Expert Signature + Per-Action Cost Coach

## Purpose

V9.9 keeps the strict V9.8 outer-fold protocol and the original frozen V9.3
Validation/Test experts. It changes what each expert exposes to the coach and what
the coach predicts.

V9.8 routed mainly by the distance between a predicted frontier value and five
close scalar candidate values. Its gain head described the best action present in
the pool rather than the action actually proposed. V9.9 removes both mismatches.

## Fixed semantic expert signature

Each specialist exposes a 21-dimensional task-aligned signature. No unaligned
fold-local latent coordinate is used.

Common fields include:

- prediction, correction, absolute correction and confidence;
- raw correction;
- optional self-predicted absolute error;
- optional region expectation and region entropy;
- optional mechanism entropy;
- explicit role/tail/risk availability masks;
- the complete five-region probability vector for RoleConditionedDLF experts;
- the complete four-mechanism probability vector for tail experts.

The Anchor receives the same schema with an explicit action embedding and zeroed
unavailable semantic fields. The prediction field always reproduces the actual
candidate output.

## Strict OOF semantic pool

For every top-level outer fold, V9.9 uses only the matching V9.2 fold-local clean,
ModDrop and CFCompatKD checkpoints. It refits the fold-local teacher committee,
Boundary, Positive, Strong Negative and Strong Positive experts, and records rich
semantic outputs on the untouched outer holdout groups.

The cache version and output directory differ from V9.8, so neither the old V9.7
pool nor the scalar-only V9.8 pool can be reused accidentally.

## Per-action cost coach

For action `k`, the coach predicts:

```text
predicted_cost[k]  ~= abs(candidate[k] - label)
predicted_scale[k] ~= uncertainty of that cost estimate
```

Training combines:

- Smooth-L1 absolute-cost regression for all five actions;
- Laplace-style scale calibration;
- relative gain regression against Anchor;
- cost-sensitive pairwise ranking;
- oracle-action cross entropy derived from predicted costs;
- soft expected selected cost.

The proposed action is the minimum predicted risk action. Its predicted gain is
computed from that same action:

```text
predicted_gain = predicted_anchor_cost - predicted_selected_cost
```

The conservative lower gain also includes the predicted cost scales. This removes
the V9.8 mismatch where a high oracle-pool gain could approve a different, wrongly
selected expert.

## Pre-registered safety profiles

Only three fixed profiles are evaluated:

```text
conservative
balanced
broad
```

They threshold the selected action's predicted gain, conservative lower gain,
predicted-cost margin, predicted scale and expert confidence. They are screened
on grouped Train OOF results before Validation. Validation only compares Anchor
with OOF-eligible profiles.

## Diagnostic Test outputs

V9.9 reports:

```text
anchor
minimum_predicted_cost_action_all
risk_adjusted_cost_action_all
semantic_cost_soft_mixture_all
semantic_cost_valid_selected
true_region_expert_policy
sample_oracle_upper_bound
four individual specialists
```

The unrestricted hard and soft routes are diagnostics. Test labels never choose a
profile or threshold.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/semantic-cost-coach-v9-9
git pull origin innovation/semantic-cost-coach-v9-9

python3 scripts/smoke_test_semantic_cost_coach_v9_9.py

GPU=0 SEED=1111 NUM_WORKERS=1 \
  bash scripts/run_mosi_semantic_cost_coach_v9_9.sh \
  2>&1 | tee mosi_semantic_cost_coach_v99_seed1111.log
```

## Output root

```text
result/semantic_cost_coach_v99/mosi/seed_1111/
```

Important files:

```text
strict_semantic_expert_pool/strict_semantic_expert_pool_v99.pth
strict_semantic_expert_pool/v99_strict_semantic_expert_pool_summary.json
strict_semantic_expert_pool/v99_strict_semantic_expert_pool.csv
v99_semantic_cost_oof_predictions.csv
v99_oof_policy_profiles.csv
v99_validation_profile_selection.csv
v99_test_summary.csv
semantic_cost_coach_v99_predictions.csv
semantic_cost_coach_v99_summary.json
```
