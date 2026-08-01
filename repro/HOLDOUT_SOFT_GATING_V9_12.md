# V9.12 Single-Holdout Soft Gating

## Purpose

V9.12 removes the five-fold deeply nested coach protocol. It uses one fixed
group-isolated Router-Train split and one unchanged expert stack.

The key invariant is:

```text
the experts that produce Router-Train outputs
are the same model instances/checkpoints used on Validation and Test
```

This directly tests whether the previous failures came from shadow-expert identity
drift.

## Split protocol

V9.12 reuses one outer fold from the completed V9.2 grouped cache.

```text
selected outer-fold development groups -> Expert-Train
selected outer-fold holdout groups     -> Router-Train
official Validation                    -> early stop and beta selection
official Test                          -> final evaluation only
```

The fold-local clean, ModDrop, and CFCompat checkpoints exclude the complete
Router-Train outer holdout. Boundary/Positive role experts and Strong-Negative/
Strong-Positive tail experts are trained once on Expert-Train. Those exact four
expert models and the exact selected Anchor are then reused unchanged on
Router-Train, Validation, and Test.

This is not five-fold coach OOF. It is one conventional held-out meta-training split.

## Soft gate

The gate produces nonnegative convex weights over:

```text
Anchor
Strong Negative
Boundary
Positive
Strong Positive
```

The prediction is:

```text
mixture = sum_k weight[k] * prediction[k]
final   = Anchor + beta * (mixture - Anchor)
```

Weights sum to one. The Anchor logit is initialized with a positive bias, so the
gate begins conservatively.

Gate training directly optimizes routed regression error with small penalties for
harm beyond Anchor, expected action cost, and unnecessary specialist mass.

Three deterministic low-capacity gate members are trained on Router-Train and
early-stopped on official Validation. Their weights are averaged. Validation
selects `beta` from the preregistered grid:

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

`beta=0` is the explicit Anchor fallback.

## What this experiment answers

The Router-Train metrics are in-sample for the gate and are diagnostics only.
Validation is the primary model-selection split.

Interpretation:

```text
Router-Train improves, Validation does not:
    the gate overfits the small Router split.

Validation improves, Test improves:
    the simpler same-expert-stack protocol works.

Validation improves, Test does not:
    Validation overfit or dataset shift remains.

Validation does not improve:
    the expert pool does not provide learnable deployable complementarity under
    this split; deep OOF was not the main blocker.
```

## Required artifacts

V9.12 requires the completed V9.2 grouped cache and its outer-fold checkpoints:

```text
result/oof_tail_residual_experts_v92/mosi/seed_1111/
  oof_cfcompat/
    nested_grouped_oof_cfcompat_cache_v92.pth
    outer_fold_0/
      clean_dlf_best_inner_valid.pth
      moddrop_evaluator_best_inner_valid.pth
      cfcompat_student_best_inner_valid.pth
```

The outer-fold directory is adjacent to the cache according to the existing V9.2
layout.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/holdout-soft-gating-v9-12
git pull origin innovation/holdout-soft-gating-v9-12

python3 scripts/smoke_test_holdout_soft_gate_v9_12.py

GPU=0 SEED=1111 OUTER_FOLD=0 NUM_WORKERS=1 \
  bash scripts/run_mosi_holdout_soft_gate_v9_12.sh \
  2>&1 | tee mosi_holdout_soft_gate_v912_seed1111.log
```

## Outputs

```text
result/holdout_soft_gate_v912/mosi/seed_1111/
  holdout_expert_stack/
    holdout_expert_pool_v912.pth
    v912_holdout_split_manifest.csv
    boundary_holdout_expert_v912.pth
    positive_holdout_expert_v912.pth
    strong_negative_holdout_expert_v912.pth
    strong_positive_holdout_expert_v912.pth
  holdout_soft_gate_member_0_v912.pth
  holdout_soft_gate_member_1_v912.pth
  holdout_soft_gate_member_2_v912.pth
  v912_validation_beta_selection.csv
  v912_test_summary.csv
  holdout_soft_gate_v912_predictions.csv
  holdout_soft_gate_v912_summary.json
```

## Primary decision

The deployable row is:

```text
holdout_soft_gate_valid_selected
```

Compare it with:

```text
anchor
holdout_soft_gate_beta_1
uniform_all_actions
true_region_semantic_upper_bound
sample_oracle_upper_bound
```

Test labels never choose an epoch, beta, model, or threshold.
