# Compatibility--Distillation Discrepancy Analysis

This analysis validates whether CFCompat's compatibility score reflects an
independent teacher--student mismatch rather than merely reproducing its own
counterfactual evaluator delta.

## Design

For validation sample (i) and condition (m\in\{LA,LV,L\}):

1. Build compatibility from the frozen ModDrop evaluator:
   `delta_eval = abs(evaluator_LAV - evaluator_m)`.
2. Calibrate `delta_eval` against the same-mode **training-only** delta
   distribution:
   `C = 1 - F_train_mid(delta_eval)`.
3. Measure an independent distillation discrepancy using a different pair:
   - teacher: clean DLF validation-best checkpoint, LAV prediction;
   - student: FixedKD validation-best checkpoint, condition-`m` prediction.
4. Report both:
   - absolute teacher--student prediction gap;
   - SmoothL1 discrepancy, matching the KD objective.
5. Aggregate by fixed compatibility quartiles and compute 95% cluster-bootstrap
   confidence intervals by sample index.

No test split is constructed or read.

## Run

```bash
git switch analysis/compatibility-gain-v1
git pull origin analysis/compatibility-gain-v1

python analyze_compatibility_kd_gap.py \
  --dataset mosi \
  --seed 1114 \
  --gpu-ids 0 \
  --num-workers 0
```

Default inputs:

- evaluator validation predictions:
  `result/missing_baseline/moddrop_benchmark_multiseed_v1/seed1114/mosi_best_valid_predictions.csv`
- train-only compatibility cache:
  `result/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/train_counterfactual_compatibility.csv`
- FixedKD checkpoint:
  `pt/missing_baseline/fixed_kd/DLF_mosi_seed1114_best.pth`
- clean teacher checkpoint:
  `pt/DLF_mosi_seed1114_best.pth`

The script generates validation predictions for FixedKD and the clean teacher
if cached copies are absent.

Outputs:

`result/analysis/compatibility_kd_gap_v1/mosi/seed1114/`

Please send back:

- `mosi_seed1114_compatibility_kd_gap_summary.csv`
- `mosi_seed1114_compatibility_kd_gap_diagnostics.json`
- the terminal quartile table and mode-wise table

## Main signal

A useful validation pattern is lower independent teacher--student discrepancy
at higher compatibility, e.g. a negative Spearman correlation between
compatibility and the independent gap. The result should be reported as
observed; monotonicity is not assumed in advance.
