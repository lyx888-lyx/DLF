# Validation-only Compatibility-stratified Distillation Analysis

This branch adds `analyze_compatibility_gain.py` for the paper mechanism analysis.

## Question

Does CFCompat help by changing the effect of uniform KD differently across compatibility regions?

The analysis compares three paired validation systems:

1. **No KD:** paired ModDrop baseline.
2. **Uniform KD:** FixedKD.
3. **Compatibility-aware KD:** CFCompat.

For each validation sample and each student condition (LA/LV/L):

1. Compute the evaluator discrepancy from the paired ModDrop validation predictions:
   `delta_m = abs(LAV_pred - m_pred)`.
2. Map `delta_m` through the **training-only** empirical delta distribution for the same mode:
   `C_m = 1 - F_train_mid(delta_m)`.
3. Compare paired absolute errors:
   - `uniform_gain = error(NoKD) - error(FixedKD)`
   - `compat_gain = error(FixedKD) - error(CFCompat)`
   - `total_gain = error(NoKD) - error(CFCompat)`
4. Report all three effects by fixed compatibility quartiles:
   `[0,.25], (.25,.5], (.5,.75], (.75,1]`.
5. Compute 95% cluster-bootstrap confidence intervals by sample index.

Positive gain always means the method on the right reduces absolute error.

No test prediction file is read. The script rejects paths containing `test`.

## Required existing outputs

For seed 1114, defaults are:

- No-KD validation predictions:
  `result/missing_baseline/moddrop_benchmark_multiseed_v1/seed1114/mosi_best_valid_predictions.csv`
- CFCompat validation predictions:
  `result/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1114/mosi_best_valid_predictions.csv`
- Train-only compatibility cache:
  `result/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/train_counterfactual_compatibility.csv`
- FixedKD validation-best checkpoint:
  `pt/missing_baseline/fixed_kd/DLF_mosi_seed1114_best.pth`

If `--fixedkd-pred` is not supplied, the script generates **validation-only**
FixedKD predictions from that checkpoint and caches them under the analysis
output directory.

## Run

```bash
python analyze_compatibility_gain.py \
  --dataset mosi \
  --seed 1114 \
  --gpu-ids 0 \
  --num-workers 0
```

Outputs are written to:

`result/analysis/compatibility_gain_v2/mosi/seed1114/`

Please send back:

- `mosi_seed1114_valid_distillation_effect_summary.csv`
- `mosi_seed1114_valid_distillation_effect_diagnostics.json`
- the printed console table

The main pattern to inspect is whether low-compatibility regions show weak or
negative uniform-KD gains while CFCompat improves over uniform KD.
