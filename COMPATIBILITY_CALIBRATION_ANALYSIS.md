# Raw Discrepancy vs. Calibrated Compatibility

This train-only analysis examines why CFCompat uses mode-wise empirical-rank
calibration instead of raw counterfactual discrepancies directly.

## Questions

1. Do LA/LV/L exhibit different raw counterfactual discrepancy scales?
2. Would a single pooled/global calibration systematically shift compatibility
   across target conditions?
3. How does mode-wise calibration map each raw discrepancy scale onto a common
   compatibility scale?

## Run

```bash
git switch analysis/compatibility-gain-v1
git pull origin analysis/compatibility-gain-v1

python analyze_compatibility_calibration.py \
  --dataset mosi \
  --seed 1114
```

The script reads only:

`result/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seed1114/train_counterfactual_compatibility.csv`

No validation or test split is accessed.

## Outputs

`result/analysis/compatibility_calibration_v1/mosi/seed1114/`

Key files:

- `mosi_seed1114_raw_discrepancy_calibration_summary.csv`
- `mosi_seed1114_raw_discrepancy_calibration_diagnostics.json`
- `mosi_seed1114_raw_discrepancy_ecdf.png`
- `mosi_seed1114_modewise_calibration_mapping.png`
- `mosi_seed1114_global_calibration_bias.png`

## Interpretation

The ECDF plot is the empirical evidence: if the LA/LV/L raw discrepancy curves
differ, the raw scales are condition dependent.

The calibration mapping plot visualizes the deterministic mode-wise mapping
from raw discrepancy to compatibility. It should be described as the mechanism
of calibration, not as independent evidence of effectiveness.

The global-calibration plot is a counterfactual diagnostic. It applies one
pooled empirical CDF to all modes. If the resulting mean compatibility differs
substantially across modes, pooled calibration would systematically favor or
suppress conditions due to raw-scale differences. Mode-wise calibration removes
that scale effect by ranking within each condition.
