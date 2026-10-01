# Pooled vs. Mode-wise Calibration Ablation

This experiment tests whether the proposed condition-wise rank calibration
improves CFCompat over a single pooled/global rank calibration.

## Compared variants

- **Pooled rank:** concatenate train-only `delta_LA`, `delta_LV`, and
  `delta_L`, compute one empirical rank distribution, and derive all
  compatibility scores from that pooled reference.
- **Mode-wise rank (ours):** compute empirical ranks independently within LA,
  LV, and L, matching the CFCompat formulation.

Everything else is held fixed: clean-teacher initialization, train split,
missing-mask RNG, optimizer, KD loss, checkpoint criterion, and seed.

This runner constructs **train and validation loaders only**. It never reads
the MOSI test split.

## Run

```bash
git switch analysis/compatibility-gain-v1
git pull origin analysis/compatibility-gain-v1

python run_calibration_ablation.py \
  --dataset mosi \
  --seed 1114 \
  --gpu-ids 0 \
  --num-workers 1
```

## Outputs

`result/analysis/calibration_ablation_v1/mosi/seed1114/`

Key files:

- `mosi_seed1114_calibration_ablation.csv`
- `mosi_seed1114_calibration_ablation_conditions.csv`
- `mosi_seed1114_calibration_ablation_epochs.csv`
- `mosi_seed1114_calibration_ablation_table.tex`

The headline table reports the macro average over the three target conditions
LA/LV/L. The condition-wise CSV is also written so any gain or degradation can
be checked separately for LA, LV, and L.

## Interpretation

If mode-wise rank improves validation metrics over pooled rank, the result
supports the claim that removing condition-dependent raw-scale bias is useful
for distillation. If the two are similar, the calibration figures can still be
used to explain the representation scale effect, but the paper should not claim
a performance advantage from mode-wise calibration itself.
