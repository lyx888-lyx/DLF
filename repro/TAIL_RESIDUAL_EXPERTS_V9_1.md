# V9.1 Bidirectional Tail Residual Experts

## Purpose

This branch does **not** retrain the successful V9 boundary and ordinary-positive
specialists. It isolates the unresolved strong-negative and strong-positive
regions and tests three candidates from the same immutable CFCompatKD anchor:

1. `shared_tail`: one residual mechanism trained on both tails;
2. `strong_negative`: a negative-tail specialist;
3. `strong_positive`: a positive-tail specialist.

The experiment decides from Validation whether the two tails should share an
expert or use separate experts. It does not assume the answer in advance.

## Main corrections relative to V9

- The anchor is read from the aligned teacher cache and is immutable in every
  loss, checkpoint metric, capability matrix, and final comparison.
- Tail specialists learn the signed residual `label - anchor`; they are not
  rewarded merely for increasing absolute sentiment strength.
- Tail curriculum weights are monotone sigmoids with no 0.25 membership floor.
  Opposite-side samples retain the anchor but do not receive a tail objective.
- An internally supervised applicability gate limits corrections outside the
  candidate's tail. Regression gradients cannot trivially force the gate open.
- Four mechanism probabilities are produced: sign repair, magnitude increase,
  magnitude decrease, and near-correct.
- Region-teacher distillation is enabled only when its Validation tail MAE beats
  the anchor by at least `--min-teacher-gain`.
- Epoch 0 is the exact anchor. A candidate checkpoint is accepted only if it
  improves its target tail while respecting global-MAE and harm-rate limits;
  otherwise that candidate safely falls back to the anchor.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/tail-residual-experts-v9-1
git pull origin innovation/tail-residual-experts-v9-1

python3 scripts/smoke_test_tail_residual_experts_v9_1.py

TEACHER_GLOB='/code/DLF/pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed*/DLF_mosi_seed*_best_valid.pth' \
GPU=0 \
SEED=1111 \
bash scripts/run_mosi_tail_residual_experts_v9_1.sh \
2>&1 | tee mosi_tail_residual_experts_v91_seed1111.log
```

## Primary outputs

Output root:

```text
result/tail_residual_experts_v91/mosi/seed_1111/
```

Important files:

- `tail_residual_experts_v91_summary.json`
- `v91_anchor_residual_diagnostics.csv`
- `v91_tail_teacher_diagnostics.csv`
- `v91_valid_tail_capability_matrix.csv`
- `v91_test_tail_capability_matrix.csv`
- `v91_anchor_gate_calibration.csv`
- `v91_test_tail_summary.csv`
- `tail_residual_experts_v91_predictions.csv`
- `<candidate>/training_history.csv`
- `<candidate>/tail_residual_expert_v91_best.pth`

## Interpretation order

1. Inspect `v91_anchor_residual_diagnostics.csv` to see whether each tail is
   dominated by sign errors, magnitude underestimation, or overestimation.
2. Inspect teacher diagnostics. A disabled tail teacher is not an error; it
   means the original five seeds do not supply a better tail target.
3. Check whether each candidate selected a learned epoch or `anchor_fallback`.
4. Compare `shared_tail` against sign-specific candidates on Validation.
5. Treat `true_region_valid_selected_tail_policy` as an analysis upper bound,
   because it uses the true Test tail region only after the policy has been fixed
   from Validation.
6. `anchor_score_gate_valid_selected` is deployable: it uses only anchor scores,
   Validation-selected thresholds, and Validation-selected shrinkage.

## Scientific gate

A useful outcome requires at least one tail region to select a non-anchor expert
on Validation. A stronger result requires the same direction of improvement on
Test. If all candidates fall back to the anchor, the implementation succeeded
but the current frozen-feature residual capacity or curriculum was insufficient.
