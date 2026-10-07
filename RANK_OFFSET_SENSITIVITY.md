# Quantile-interpolation sensitivity: why use alpha = 0.5?

CFCompat calibrates each condition-wise counterfactual discrepancy through an
empirical tied interval. For sample i under target condition m, let

- L_i^m be the number of training discrepancies strictly smaller than delta_i^m;
- T_i^m be the size of the tied group containing delta_i^m;
- N_m be the number of training samples under condition m.

The generalized empirical quantile is

```text
q_i^m(alpha) = (L_i^m + alpha * T_i^m) / N_m
C_i^m(alpha) = 1 - q_i^m(alpha),     alpha in [0, 1].
```

The default alpha = 0.5 is the midpoint configuration. The sensitivity sweep is
now the symmetric set

```text
{0.00, 0.25, 0.50, 0.75, 1.00}
```

so that the experiment explicitly covers both empirical-interval boundaries,
the two quarter positions, and the midpoint.

## Why alpha = 0.5?

The midpoint follows from the symmetry condition inside a tied empirical
interval: the assigned position is equidistant from the interval's lower and
upper boundaries. It is therefore a principled default rather than a value
chosen from test performance.

For a fixed sample,

```text
C_i^m(alpha) - C_i^m(0.5)
    = (0.5 - alpha) * T_i^m / N_m.
```

Hence the effect of alpha depends on the tied-group size T_i^m. When ties are
small relative to N_m, the compatibility perturbation is correspondingly
small.

## Endpoint handling

At alpha = 0 or alpha = 1, samples at an empirical boundary may receive exact
compatibility values of 1 or 0. This is intentional for the endpoint
sensitivity experiment. The runner therefore permits compatibility in the
closed interval [0, 1] locally for this analysis. The main CFCompat pipeline
remains unchanged.

A zero compatibility weight is valid in the normalized KD objective: it simply
removes that sample's teacher contribution from the weighted numerator. The
existing epsilon in the denominator maintains numerical stability.

## Protocol

- dataset: CMU-MOSI
- seed: 1114
- Train is used for optimization and Validation selects the checkpoint
- after the validation-best checkpoint is frozen, Test is evaluated once for final reporting
- same clean teacher, student initialization, optimizer, missing-mask RNG,
  KD objective, and validation checkpoint criterion for every alpha
- target metrics are macro-averaged over LA/LV/L

Because the mathematical definition of alpha has been updated from the old
rank-offset form to tied-interval interpolation, rerun all five settings rather
than reusing the previous 0.25/0.50/0.75 rows. The five alpha values are fixed
before Test is accessed; Test is not used to choose alpha or checkpoints.

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

For a one-epoch smoke test first:

```bash
python run_rank_offset_sensitivity.py \
  --dataset mosi \
  --seed 1114 \
  --gpu-ids 0 \
  --num-workers 1 \
  --max-epochs 1 \
  --overwrite
```

## Outputs

Results are written to:

```text
result/analysis/quantile_interpolation_sensitivity_v3/mosi/seed1114/
```

Key files:

- `mosi_seed1114_quantile_interpolation_sensitivity.csv`
- `mosi_seed1114_quantile_interpolation_sensitivity_conditions.csv`
- `mosi_seed1114_quantile_interpolation_sensitivity_epochs.csv`
- `mosi_seed1114_quantile_interpolation_sensitivity_table.tex`

Checkpoints are written under:

```text
pt/analysis/quantile_interpolation_sensitivity_v3/mosi/seed1114/
```

The generated LaTeX table uses Test macro metrics over LA/LV/L for the
paper-facing rows, while the CSV retains Validation metrics and the
validation-selected epoch for auditability. The table uses the paper terminology
“quantile interpolation coefficient alpha” rather than the obsolete “rank offset”.

Report the observed results directly. The purpose of the sweep is to test the
robustness of the symmetry-derived midpoint configuration, not to select alpha
from validation or test outcomes after the fact.
