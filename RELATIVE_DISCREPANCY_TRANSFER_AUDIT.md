# CFCompat relative-discrepancy and transfer audit (Validation only)

Source: frozen main-protocol branch analysis/compatibility-gain-v1.
No model retraining or checkpoint changes are introduced in this branch.

## Hypotheses (not presupposed to be true)

H1 **Condition-specific raw gap scales:** near-equal evaluator counterfactual
discrepancies can correspond to different TRAIN-referenced compatibility ranks
under LA/LV/L. This is a descriptive property, not yet a KD improvement claim.

H2 **Transfer relevance of condition-wise normalization:** compared with pooled
normalization, mode-wise compatibility may show a better association with
(a) independent Teacher--Uniform-student prediction discrepancy, or (b) actual
gain of Uniform KD versus No KD. Both can fail: the script reports the signs
and confidence intervals without forcing the narrative.

H3 **Fine-grained sentiment intensity:** among validation sample-views where
both Uniform KD and CFCompat correctly classify the polarity, CFCompat may
have lower continuous-label absolute error, especially on higher |label| cases.

## Existing files reused (MOSI seed 1114)

- NoKD/frozen Evaluator Valid predictions:
  result/missing_baseline/moddrop_benchmark_multiseed_v1/seed1114/mosi_best_valid_predictions.csv
- CFCompat Valid predictions:
  result/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1114/mosi_best_valid_predictions.csv
- train-only evaluator cache:
  result/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/train_counterfactual_compatibility.csv
- Uniform KD Valid predictions: generate or reuse using the existing
  analyze_compatibility_kd_gap.py script:
  result/analysis/compatibility_kd_gap_v1/mosi/seed1114/mosi_seed1114_fixedkd_valid_predictions.csv
- Clean Teacher Valid predictions (optional but strongly recommended), generated
  by the same existing script:
  result/analysis/compatibility_kd_gap_v1/mosi/seed1114/mosi_seed1114_clean_teacher_valid_predictions.csv

These are predictions from frozen checkpoints, not training-time compatibility
gates. For validation sample j and condition m, the score is computed against
the *train-only* mode-m reference distribution.

## Commands

~~~bash
cd /code/DLF
git fetch origin
git switch analysis/cfcompat-relative-gap-transfer-audit-v1
git pull origin analysis/cfcompat-relative-gap-transfer-audit-v1

# Minimal synthetic tests (CPU only)
python -m unittest discover -s tests -p 'test_relative_discrepancy_transfer.py' -v

# Run once to create Uniform KD and clean Teacher Valid prediction CSVs,
# if they do not already exist:
CUDA_VISIBLE_DEVICES=0 python analyze_compatibility_kd_gap.py \
  --dataset mosi --seed 1114 --gpu-ids 0 --num-workers 0 --no-plot

# Main audit (CPU, frozen CSV only)
python analyze_relative_discrepancy_transfer.py \
  --dataset mosi --seed 1114 \
  --uniform-valid result/analysis/compatibility_kd_gap_v1/mosi/seed1114/mosi_seed1114_fixedkd_valid_predictions.csv \
  --teacher-valid result/analysis/compatibility_kd_gap_v1/mosi/seed1114/mosi_seed1114_clean_teacher_valid_predictions.csv
~~~

To run after finishing seed 1111, repeat with --seed 1111 and change the
two explicit paths to their seed1111 equivalents. DO NOT search seeds or
thresholds based on preferred downstream results.

The initial predeclared matched-pair tolerance is 5% of the pooled TRAIN
discrepancy IQR. The rank-separation threshold is 0.25. A matched pair is
restricted to the *same validation sample* in two distinct input conditions:
it must have nearly equal raw evaluator discrepancies but different
condition-wise ranks. This safeguards the target label and utterance identity.
Matching does not control for different input modalities and cannot establish
causality. Empty match sets are a valid negative outcome.

## Outputs

The output directory is:
result/analysis/relative_gap_transfer_v1/mosi/seed1114/

- *_valid_relative_gap_rows.csv: each Valid sample × LA/LV/L, predictions,
  relative/raw/pool calibration, signed KD and CFCompat gains, optional
  independent teacher--student gaps
- *_same_sample_matched_gap_pairs.csv: matched near-equal raw gap cases;
  sorted transparently by rank separation, not error improvement
- *_modewise_vs_pooled_rank_correlation.csv: pooled and per-mode Spearman
  associations with actual KD gain and independent teacher--student gap;
  paired sample-cluster bootstrap for the across-modes difference
- *_polarity_correct_intensity.csv: conditional Uniform KD versus CFCompat
  MAE for jointly correct-polarity predictions, by |sentiment| group
- *_fixed_gap_deciles_by_mode.csv: deciles determined ONLY by pooled Train data
- *_relative_gap_motivation.{png,pdf}: data-grounded two-panel illustration
- *_protocol_and_diagnostics.json: exact sources, criteria, summary, warnings

## How to interpret

Supported H1: fixed raw-gap regions produce substantially different
condition-relative ranks; the same-sample matched-pair file has enough examples.
An isolated pair is not sufficient to claim widespread scale mismatch.

Supported H2: after pre-specified pooled vs modewise comparisons, association
with the independent gap or Uniform KD's signed benefit improves across target
conditions and (ideally) has a positive confidence bound, with replication in
independent seeds. A non-significant difference means the motivation is not
established. Do NOT claim raw discrepancy lacks information: within each mode,
rank and delta determine each other up to ties.

Supported H3: jointly polarity-correct subsets have lower CFCompat absolute
errors (especially on strong labels), with uncertainty estimates that exclude
zero. This supports an association of the complete CFCompat training strategy
with better intensity prediction, not direct proof of a unique causal mechanism
or of superiority to CMAD (which requires matched CMAD predictions).

Analysis must be reported as observational and is never allowed to tune
training weights, choose checkpoints, search Test cases or replace the
original task protocol. No Test file is accepted by this script.
