# Validation-only Compatibility Gain Analysis

This branch adds `analyze_compatibility_gain.py` for the paper mechanism analysis.

## Question

Does the train-calibrated CFCompat score predict where compatibility-guided distillation helps?

For each validation sample and each student condition (LA/LV/L):

1. Compute the evaluator discrepancy from the paired ModDrop validation predictions:
   `delta_m = abs(LAV_pred - m_pred)`.
2. Map `delta_m` through the **training-only** empirical delta distribution for the same mode:
   `C_m = 1 - F_train_mid(delta_m)`.
3. Compare paired absolute errors:
   `gain = |y - pred_ModDrop| - |y - pred_CFCompat|`.
   Positive gain means CFCompat improves the sample.
4. Report gain by fixed compatibility quartiles [0,.25], (.25,.5], (.5,.75], (.75,1].
5. Compute 95% cluster-bootstrap confidence intervals by sample index.

No test prediction file is read; the script rejects paths containing test.

## Required existing outputs

For seed 1114, defaults are:

- `result/missing_baseline/moddrop_benchmark_multiseed_v1/seed1114/mosi_best_valid_predictions.csv`
- `result/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1114/mosi_best_valid_predictions.csv`
- `result/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/train_counterfactual_compatibility.csv`

## Run

```bash
python analyze_compatibility_gain.py --dataset mosi --seed 1114
```

Outputs are written to:

`result/analysis/compatibility_gain_v1/mosi/seed1114/`

Please send back:

- `mosi_seed1114_valid_compatibility_gain_summary.csv`
- `mosi_seed1114_valid_compatibility_gain_diagnostics.json`
- the printed console table

The PNG is optional for deciding whether the trend is paper-worthy.
