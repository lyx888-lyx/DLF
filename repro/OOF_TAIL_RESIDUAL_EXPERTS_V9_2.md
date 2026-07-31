# V9.2 Nested Grouped OOF CFCompatKD Tail Experts

## Why this experiment exists

V9.1 proved that the full-train CFCompatKD anchor has opposite tail residual
biases on its own training samples versus Validation/Test. Therefore the signed
residual `label - full_train_anchor(train_sample)` is not a valid supervision
signal for a deployable tail expert.

V9.2 replaces that signal with nested, group-cross-fitted CFCompatKD predictions
and fusion features. Every training sample is predicted by a CFCompatKD student
whose entire training and checkpoint-selection pipeline excludes that sample's
source video/conversation group.

## Leakage boundary

For each outer fold:

1. all segments from the same inferred source-video group stay together;
2. the outer holdout groups are removed;
3. the remaining groups are split into inner train and inner validation groups;
4. a clean DLF teacher is trained on inner train and selected on inner validation;
5. a ModDrop evaluator is initialized from that clean checkpoint, trained on
   inner train, and selected on inner validation;
6. counterfactual compatibility ranks are fitted only on inner train;
7. a CFCompatKD student is initialized from the clean checkpoint, trained on
   inner train, and selected on inner validation;
8. only then is the untouched outer holdout predicted and its final fusion
   feature cached.

The five existing full-data CFCompatKD checkpoints are **not** used anywhere in
OOF construction. They are used only after the OOF cache is complete to choose
the full-data deployable anchor from Validation and extract Validation/Test
features.

## Cost

The default is three outer folds. Each fold trains three models, so the OOF stage
contains nine validation-selected training runs. It is intentionally expensive.
The output is resumable: a completed `outer_holdout_cache.pth` skips that fold on
the next invocation.

## Files

- `build_grouped_oof_cfcompat_v9_2.py`: nested OOF builder.
- `train_oof_tail_residual_experts_v9_2.py`: trains the three low-capacity tail
  heads from OOF features and residuals.
- `trains/singleTask/grouped_oof_cfcompat_v92.py`: group splitting and all three
  fold-training stages.
- `trains/singleTask/model/CachedTailResidualHeadV92.py`: frozen-anchor residual
  head.
- `trains/singleTask/oof_tail_residual_v92.py`: loss and diagnostics.
- `trains/singleTask/oof_tail_residual_system_v92.py`: Validation selection and
  final evaluation.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/oof-tail-residual-experts-v9-2
git pull origin innovation/oof-tail-residual-experts-v9-2

python3 scripts/smoke_test_oof_tail_residual_experts_v9_2.py

TEACHER_GLOB='/code/DLF/pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed*/DLF_mosi_seed*_best_valid.pth' \
GPU=0 \
SEED=1111 \
OUTER_FOLDS=3 \
bash scripts/run_mosi_oof_tail_residual_experts_v9_2.sh \
2>&1 | tee mosi_oof_tail_residual_experts_v92_seed1111.log
```

## Important outputs

Output root:

```text
result/oof_tail_residual_experts_v92/mosi/seed_1111/
```

OOF construction:

- `oof_cfcompat/v92_nested_group_manifest.csv`
- `oof_cfcompat/nested_grouped_oof_cfcompat_cache_v92.pth`
- `oof_cfcompat/nested_grouped_oof_cfcompat_predictions_v92.csv`
- `oof_cfcompat/nested_grouped_oof_cfcompat_summary_v92.json`
- `oof_cfcompat/outer_fold_*/clean_history.csv`
- `oof_cfcompat/outer_fold_*/moddrop_history.csv`
- `oof_cfcompat/outer_fold_*/cfcompat_history.csv`

Tail expert stage:

- `v92_oof_anchor_residual_diagnostics.csv`
- `v92_oof_valid_direction_agreement.csv`
- `v92_valid_tail_capability_matrix.csv`
- `v92_test_tail_capability_matrix.csv`
- `v92_gate_calibration.csv`
- `v92_test_tail_summary.csv`
- `oof_tail_residual_experts_v92_predictions.csv`
- `oof_tail_residual_experts_v92_summary.json`

## Interpretation order

1. The engineering audit must pass.
2. Check `v92_oof_valid_direction_agreement.csv`. OOF train and Validation tail
   mean residuals should preferably have the same direction.
3. Inspect OOF MAE and per-fold inner-validation histories. A fold with a broken
   or degenerate model invalidates the cache.
4. Check whether `shared_tail`, `strong_negative`, or `strong_positive` selects a
   learned epoch rather than the anchor fallback.
5. Expert assignment, gate thresholds, and shrinkage are selected using
   Validation only.
6. The true-region tail policy and sample oracle are diagnostic upper bounds.

## Resume behavior

Rerunning the same command reuses completed outer-fold caches. Use
`--no-resume` on `build_grouped_oof_cfcompat_v9_2.py` only when intentionally
retraining every fold.
