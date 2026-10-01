# Rank-offset sensitivity: why use 0.5?

CFCompat uses the empirical-rank transform

[
q_i^m = rac{r_i^m - 0.5}{N_m}, qquad C_i^m = 1 - q_i^m.
]

This experiment generalizes the offset to

[
q_i^m(\alpha) = rac{r_i^m - \alpha}{N_m}, qquad
C_i^m(\alpha) = 1 - q_i^m(\alpha),
]

and evaluates \(\alpha\in\{0.10,0.25,0.50,0.75,0.90\}\).

## Why 0.5?

The value 0.5 is the midpoint rank correction. It places each empirical rank
at the center of its rank interval and keeps both the quantile and
compatibility strictly inside (0,1) for finite training sets.

It should therefore be treated as a standard symmetric convention, not as a
performance-tuned hyperparameter.

## Expected sensitivity

For a fixed rank and training size \(N_m\),

[
C_i^m(\alpha)-C_i^m(0.5)=rac{\alpha-0.5}{N_m}.
]

For MOSI, \(N_m=1284\), so even changing \(\alpha\) from 0.5 to 0.1 shifts
every compatibility value by only about \(-3.12\times10^{-4}\). Because the
KD loss also normalizes by the sum of compatibility weights, performance should
be largely insensitive to this offset.

The useful result is therefore robustness, not necessarily that 0.5 wins every
metric.

## Protocol

- seed: 1114
- train and validation only
- no MOSI test access
- same clean teacher, student initialization, optimizer, missing-mask RNG,
  KD objective, and validation checkpoint criterion for every alpha
- target metrics are macro-averaged over LA/LV/L

## Run

```bash
git switch analysis/compatibility-gain-v1
git pull origin analysis/compatibility-gain-v1

python run_rank_offset_sensitivity.py \
  --dataset mosi \
  --seed 1114 \
  --gpu-ids 0 \
  --num-workers 1
```

## Outputs

`result/analysis/rank_offset_sensitivity_v1/mosi/seed1114/`

Key files:

- `mosi_seed1114_rank_offset_sensitivity.csv`
- `mosi_seed1114_rank_offset_sensitivity_conditions.csv`
- `mosi_seed1114_rank_offset_sensitivity_epochs.csv`
- `mosi_seed1114_rank_offset_sensitivity_table.tex`

Send back the main CSV and the terminal summary after the run.
