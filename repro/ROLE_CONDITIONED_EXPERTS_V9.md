# V9 Role-Conditioned Complementary Distillation

Branch: `innovation/role-conditioned-experts-v9`

## Scientific objective

V9 tests a stronger hypothesis than ordinary seed ensembling:

> Five strong multimodal experts can be deliberately trained to have different,
> validation-verifiable competence envelopes over fixed MOSI/MOSEI sentiment
> regions, while retaining acceptable global performance.

The five roles are shared across MOSI and MOSEI because both use the continuous
`[-3, 3]` sentiment scale:

| Index | Role | Label interval |
|---:|---|---|
| 0 | strong_negative | `y < -1.5` |
| 1 | negative | `-1.5 <= y < -0.5` |
| 2 | boundary | `-0.5 <= y <= 0.5` |
| 3 | positive | `0.5 < y <= 1.5` |
| 4 | strong_positive | `y > 1.5` |

## Why this is not a simple weighted-loss implementation

Each expert is initialized from exactly the same validation-best strong
multimodal checkpoint **and the same new-head random seed**. This removes random
initialization as an explanation for the capability matrix. Specialization is
then induced by five coordinated mechanisms:

1. **Validation-fitted role Teacher targets.** A global simplex is fitted on
   Validation. Each role then receives a region-specific Teacher simplex,
   selected by deterministic internal folds and shrunk toward the global
   simplex when the region is sparse.
2. **Overlapping role curriculum.** Every expert sees every training sample.
   A Gaussian role membership plus a nonzero floor prevents hard partitioning
   and catastrophic loss of general ability.
3. **Role-specific distillation and mechanism losses.** Tail experts preserve
   sign and intensity ordering; ordinary polarity experts emphasize sign;
   the boundary expert emphasizes calibrated distance from zero.
4. **Explicit gain margin.** In the assigned region, an expert is trained to
   improve on the frozen anchor rather than merely copy it.
5. **Global competence protection.** Global supervised loss, global committee
   distillation, outside-region anchor retention, bounded residuals, and a
   validation global-degradation penalty prevent artificial specialization by
   intentionally damaging other regions.

Every expert also produces an independent five-region probability distribution
and a stop-gradient predicted absolute-error score. The risk head cannot alter
shared expert representations.

## Leakage controls

- Teacher checkpoint choice: Validation only.
- Global/role Teacher weights: Validation only, with deterministic hash-fold CV.
- Expert checkpoint: Validation objective only.
- Category-coach temperature, risk coefficient, and anchor shrinkage: Validation
  only.
- Test labels: final metrics and explicitly named oracle diagnostics only.
- Train loader: reproducible epoch-dependent shuffle.
- Validation/Test loaders: `shuffle=False`, `drop_last=False`.
- All cross-model collection: aligned by unique sample ID into Teacher-cache
  order.

## First run: MOSI single seed

Use the same strong CFCompatKD checkpoints that were used for the V7 committee.
The glob must resolve to at least three strong multimodal checkpoints.

```bash
cd /code/AAAI/DLF
git fetch origin
git checkout innovation/role-conditioned-experts-v9
git pull origin innovation/role-conditioned-experts-v9

python3 scripts/smoke_test_role_conditioned_experts_v9.py

TEACHER_GLOB='/absolute/path/to/cfcompatkd/mosi/seed_*/best.pth' \
GPU=0 SEED=1111 \
bash scripts/run_mosi_role_conditioned_experts_v9.sh \
  2>&1 | tee mosi_role_conditioned_experts_v9_seed1111.log
```

Equivalent explicit invocation:

```bash
python3 train_role_conditioned_experts_v9.py \
  --dataset mosi \
  --seed 1111 \
  --gpu 0 \
  --teacher-glob '/absolute/path/to/cfcompatkd/mosi/seed_*/best.pth'
```

Do not enable `--non-bert-epochs` in the first run. The default schedule is a
four-epoch heads stage followed by a conservative tail-only stage.

## MOSEI run

```bash
TEACHER_GLOB='/absolute/path/to/cfcompatkd/mosei/seed_*/best.pth' \
GPU=0 SEED=1111 \
bash scripts/run_mosei_role_conditioned_experts_v9.sh \
  2>&1 | tee mosei_role_conditioned_experts_v9_seed1111.log
```

## Key outputs

Root: `result/role_conditioned_experts_v9/<dataset>/seed_<seed>/`

- `role_conditioned_experts_v9_summary.json`
- `role_conditioned_experts_v9_predictions.csv`
- `v9_role_teacher_weights.json`
- `v9_role_teacher_cv.csv`
- `v9_valid_capability_matrix.csv`
- `v9_test_capability_matrix.csv`
- `v9_category_coach_calibration.csv`
- `v9_test_summary.csv`
- `<role>/role_conditioned_expert_v9_best.pth`
- `<role>/training_history.csv`

## Decision sequence

The first question is not whether the category coach beats V7.1. Check these in
order:

1. **Engineering validity:** audit passes; IDs are unique; probabilities and
   simplex/coach weights sum to one; no non-finite values.
2. **Global retention:** each role's selected Validation `global_delta` should
   normally remain within `+0.02`, preferably within `+0.01`.
3. **Role improvement:** each role's Validation `role_gain` should be positive.
4. **Specialization matrix:** the designated expert should be the best expert in
   its own Validation region for at least three of five regions in a promising
   first run.
5. **Actionable upper bound:** `true_region_designated_oracle` should clearly
   improve over the anchor and preferably over fixed global committee results.
6. **Deployable recovery:** the Validation-selected category coach should recover
   part of that gain on Test without an unacceptable harm rate.

A specialization claim requires multi-seed stability. A single-seed diagonal
capability matrix is only diagnostic evidence.

## Audit

```bash
python3 scripts/audit_role_conditioned_experts_v9.py \
  --root result/role_conditioned_experts_v9/mosi/seed_1111
```

The audit separates engineering correctness from scientific success. It may
print `ENGINEERING AUDIT PASSED` while also printing
`SPECIALIZATION SIGNAL: NOT YET ESTABLISHED`; that is a valid negative result
and must not be hidden.

A stricter optional gate is:

```bash
python3 scripts/audit_role_conditioned_experts_v9.py \
  --root result/role_conditioned_experts_v9/mosi/seed_1111 \
  --require-designated-wins 3
```

## Important interpretation constraints

- `true_region_designated_oracle` uses Test labels only as an analysis upper
  bound; it is not deployable and must never be reported as the final method.
- `sample_oracle_upper_bound` is an even looser analysis ceiling.
- A small global simplex improvement alone does not establish role
  specialization. The validation capability matrix and designated-role wins are
  the primary evidence.
- If the new experts fail to specialize, preserve the result. Do not change
  region boundaries or loss weights after examining Test.
