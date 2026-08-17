# Sentiment-Region & Missing-Modality Robustness Analysis v1

## Goal

This branch is a **post-hoc analysis branch**. It does not train models, select checkpoints, search blend weights, or change the frozen method. Its purpose is to answer two paper-level questions from already-frozen prediction files:

1. **Where does the full-modality gain come from?**
   - Is the improvement concentrated around near-neutral / mild sentiment, moderate sentiment, or extreme sentiment?
   - Does the method improve fine-grained intensity estimation (MAE / Corr / Acc-7 / Acc-5) without materially changing polarity classification (Acc-2 / F1)?

2. **How does performance degrade as modalities disappear?**
   - Does the baseline show a cliff-like degradation from `LAV -> one missing -> L`?
   - Does Ours degrade more gracefully?
   - Which sentiment regions are most vulnerable to missing modalities, and where does Ours preserve the largest advantage?

No claim such as “near-neutral samples are 60% of the data” is assumed in advance. The script reports the actual sample shares.

## Frozen analysis bins

The primary sentiment-intensity bins are fixed before running the cross-dataset analysis:

- `near_neutral`: `|y| <= 0.5`
- `mild`: `0.5 < |y| <= 1.5`
- `moderate`: `1.5 < |y| <= 2.5`
- `extreme`: `|y| > 2.5`

The same bins must be reused on MOSI and MOSEI. Do not redefine bins after inspecting MOSEI results.

A secondary signed-bin table is also produced only for interpretation; it does not replace the frozen primary bins.

## Expected prediction format

The preferred input is the repository-wide prediction format already used by the CFCompat ensemble utilities:

```text
sample_index,sample_id,label,LAV_pred,LA_pred,LV_pred,L_pred
```

`sample_id` is optional for this analysis. At minimum, the file must contain:

```text
sample_index,label,LAV_pred
```

Missing-mode analysis is performed only for modes available in **both** baseline and Ours.

The script can either read a final Ours CSV directly, or construct a fixed two-component blend offline:

```text
Ours = (1 - lambda) * left + lambda * right
```

This is only a replay/composition convenience. `lambda` must already be frozen elsewhere; this analysis must not search it.

## Test-use guard

The script is offline: it reads prediction CSVs only and never constructs a dataset loader or runs a model forward pass.

For `--split test`, the caller must explicitly pass:

```text
--allow-test-analysis
```

This guard is there to make post-hoc Test analysis conscious and auditable. It does **not** make a historically accessed Test pristine.

## Outputs

The analysis writes:

- `overall_mode_metrics.csv`
  - exact DLF-style metrics for each available mode and method
  - degradation relative to each method's own `LAV` performance
- `intensity_bin_metrics.csv`
  - sample count/share, MAE, bias, RMSE, within-0.5/1.0 rate, Acc-7/5/2, F1, Corr
- `intensity_bin_gain.csv`
  - Ours-vs-baseline MAE gain and win/tie/loss rate in every mode/bin
- `signed_bin_metrics.csv`
  - secondary signed-region analysis
- `sample_error_gain.csv`
  - per-sample absolute-error gain: `|e_baseline| - |e_ours|`
- `missing_level_metrics.csv`
  - `LAV`, mean of one-missing modes (`LA/LV`), and `L`
- `robustness_degradation.csv`
  - absolute and relative degradation from `LAV`
- `summary.json`
- `summary.md`
- `fig_label_distribution.png`
- `fig_lav_mae_by_intensity.png`
- `fig_error_gain_vs_label.png`
- `fig_missing_degradation_mae.png`
- `fig_mae_gain_heatmap.png`

## Interpretation order

Use the outputs in this order:

1. `fig_label_distribution` / bin shares: establish where the data are concentrated.
2. `fig_lav_mae_by_intensity` + `intensity_bin_gain`: establish where the full-modality gain comes from.
3. `fig_error_gain_vs_label`: verify the gain is not caused by a few isolated examples.
4. `fig_missing_degradation_mae`: compare cliff-like vs graceful degradation.
5. `fig_mae_gain_heatmap`: identify which sentiment regions become fragile under missing modalities.
6. Repeat **the identical protocol** on MOSEI for generalization.

## Example

Direct final prediction CSVs:

```powershell
python .\analyze_sentiment_region_missing_robustness.py `
  --dataset mosi `
  --split test `
  --baseline-predictions <DLF_PREDICTIONS.csv> `
  --ours-predictions <OURS_PREDICTIONS.csv> `
  --allow-test-analysis
```

Offline fixed blend from two already-frozen components:

```powershell
python .\analyze_sentiment_region_missing_robustness.py `
  --dataset mosi `
  --split valid `
  --baseline-predictions <DLF_PREDICTIONS.csv> `
  --ours-left-predictions <RAW5_PREDICTIONS.csv> `
  --ours-right-predictions <AUX_PREDICTIONS.csv> `
  --blend-lambda 0.5
```

The second form does **not** authorize weight search. It evaluates only the supplied, already-frozen `lambda`.
