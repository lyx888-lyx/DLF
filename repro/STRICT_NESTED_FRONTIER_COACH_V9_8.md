# V9.8 Strict Nested Frontier Coach

## Purpose

V9.8 keeps the original frozen V9.3 expert pool for Validation/Test deployment.
It replaces only the Train OOF shadow pool that supervises the frontier coach.

V9.7 incorrectly initialized fold-local Boundary/Positive experts from historical
checkpoints trained on the complete Train split. V9.8 removes that path entirely.
No historical full-Train teacher checkpoint or teacher cache is used while
constructing the shadow pool.

## Strict outer-fold protocol

V9.8 reuses the already saved V9.2 outer-fold upstream checkpoints:

```text
oof_cfcompat/outer_fold_k/clean_dlf_best_inner_valid.pth
oof_cfcompat/outer_fold_k/moddrop_evaluator_best_inner_valid.pth
oof_cfcompat/outer_fold_k/cfcompat_student_best_inner_valid.pth
```

These checkpoints were trained on the matching outer fold's `inner_train` groups,
selected on disjoint `inner_valid` groups, and never saw the complete
`outer_holdout` groups.

For each outer fold V9.8:

1. loads only those three fold-local upstream checkpoints;
2. predicts a fold-local teacher committee on the development groups;
3. chooses the fold-local anchor and fits global/role teacher weights using only
   the fold's inner validation groups;
4. trains Boundary and Positive with the original `RoleConditionedDLF`
   architecture on development groups;
5. trains Strong Negative and Strong Positive with the original
   `CachedTailResidualHeadV92` architecture on fold-local CFCompat features;
6. predicts the untouched outer holdout groups;
7. assembles the 1284-sample strict shadow pool.

The final Validation/Test predictions still use the original V9.3 frozen experts.
The strict shadow pool is only coach supervision.

## Leakage controls

The pool records every outer fold's:

- inner-train, inner-validation and holdout groups;
- exact upstream checkpoint paths;
- SHA-256 checkpoint hashes;
- fold-local teacher validation MAEs;
- selected anchor teacher;
- global and role teacher weights.

The audit rejects:

- any overlap among inner-train, inner-validation and holdout groups;
- an upstream checkpoint outside its matching `outer_fold_k` directory;
- changed checkpoint hashes;
- reuse of historical full-Train teachers;
- an implausibly low OOF sample-oracle MAE below `0.20` by default.

The last condition is a leakage sentinel rather than a theoretical bound. It can
be disabled explicitly with:

```bash
--minimum-plausible-oof-oracle-mae 0
```

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/strict-nested-frontier-coach-v9-8
git pull origin innovation/strict-nested-frontier-coach-v9-8

python3 scripts/smoke_test_strict_nested_frontier_coach_v9_8.py

GPU=0 SEED=1111 NUM_WORKERS=1 \
  bash scripts/run_mosi_strict_nested_frontier_coach_v9_8.sh \
  2>&1 | tee mosi_strict_nested_frontier_coach_v98_seed1111.log
```

The V9.2 fold-local upstream checkpoints must still exist. V9.8 does not reuse
V9.7 pool caches and writes to a separate output root.

## Output root

```text
result/strict_nested_frontier_coach_v98/mosi/seed_1111/
```

Important files:

```text
strict_nested_frontier_pool/strict_nested_v93_frontier_pool_v98.pth
strict_nested_frontier_pool/v98_strict_nested_frontier_pool_summary.json
strict_nested_frontier_pool/v98_strict_nested_frontier_pool.csv
v98_frontier_coach_oof_predictions.csv
v98_oof_policy_profiles.csv
v98_validation_profile_selection.csv
v98_test_summary.csv
strict_nested_frontier_coach_v98_predictions.csv
strict_nested_frontier_coach_v98_summary.json
```

The underlying coach implementation is shared with V9.7, so legacy-named V9.7
artifacts are also retained in the V9.8 output directory for traceability. V9.8
aliases are the preferred files.
