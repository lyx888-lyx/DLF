# Calibration Design Ablation

This experiment compares six pre-declared compatibility-calibration strategies
for CFCompat on CMU-MOSI.

## Compared variants

Two calibration scopes are crossed with three mapping functions:

| Scope | Mapping | Description |
|---|---|---|
| Pooled | Min-Max | One global min/max from all LA/LV/L train discrepancies |
| Pooled | Gaussian-CDF | One global Gaussian fit from all LA/LV/L train discrepancies |
| Pooled | Empirical-CDF | One global empirical midpoint CDF |
| Mode-wise | Min-Max | Separate train min/max for LA, LV, and L |
| Mode-wise | Gaussian-CDF | Separate Gaussian fits for LA, LV, and L |
| Mode-wise | Empirical-CDF | Separate empirical midpoint CDFs (CFCompat / Ours) |

Everything else is held fixed: clean-teacher initialization, train split,
missing-mask RNG, optimizer, KD loss, checkpoint criterion, and seed.

The original CFCompat calibration is exactly
`modewise_empirical`. The implementation checks that it matches the audited
cached empirical-rank compatibility.

## Validation selection and Test reporting

Training and checkpoint selection use **Train + Validation only**. The Test
split is not constructed during optimization.

When `--evaluate-test` is supplied, the script first freezes the
validation-best checkpoint for each pre-declared variant. Only then is the Test
loader constructed, and that fixed checkpoint is evaluated once for aggregate
reporting. Test metrics are never used for early stopping, hyperparameter
selection, or calibration fitting.

Because the final paper table is intended to report Test performance, run all
six fixed variants with the same command and do not alter the configurations
after inspecting Test results.

## Smoke test without Test access

```bash
git switch analysis/compatibility-gain-v1
git pull origin analysis/compatibility-gain-v1

python run_calibration_ablation.py \
  --dataset mosi \
  --seed 1114 \
  --gpu-ids 0 \
  --num-workers 1 \
  --max-epochs 1 \
  --overwrite
```

This checks that all six variants enter training, while leaving Test untouched.

## Formal run with final Test reporting

```bash
python run_calibration_ablation.py \
  --dataset mosi \
  --seed 1114 \
  --gpu-ids 0 \
  --num-workers 1 \
  --evaluate-test
```

If a previous partial `calibration_ablation_v2` run already created
checkpoints, either remove only those partial v2 outputs or rerun intentionally
with `--overwrite`.

## Outputs

Results are written under:

`result/analysis/calibration_ablation_v2/mosi/seed1114/`

Key files:

- `mosi_seed1114_calibration_ablation.csv`
- `mosi_seed1114_calibration_ablation_conditions.csv`
- `mosi_seed1114_calibration_ablation_epochs.csv`
- `mosi_seed1114_calibration_ablation_table.tex`

Checkpoints are written under:

`pt/analysis/calibration_ablation_v2/mosi/seed1114/`

The headline CSV contains both validation metrics and, when requested, final
Test metrics. For the paper-facing calibration table, use the
`TestMacro_*` columns, which macro-average the three target conditions
LA/LV/L. `TestLAV_*` is also retained for diagnostics.

The condition-wise CSV contains separate Validation/Test rows for LA, LV, and
L so the source of any gain or degradation can be inspected directly.

## Interpretation

This experiment tests two design choices independently:

1. whether discrepancies should be calibrated globally or separately by target
   condition;
2. whether calibration should rely on raw scale (Min-Max), a parametric
   distributional assumption (Gaussian-CDF), or the proposed non-parametric
   empirical CDF.

The paper should report the observed trade-offs rather than select calibration
variants from Test performance.
