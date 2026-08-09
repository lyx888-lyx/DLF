# ADPEP / Hybrid Missing-Modality Benchmark v1

## Goal

Before starting another innovation line, measure whether the two previously
successful MOSI methods, ADPEP-All and the 0.6995 anchored-complementarity
Hybrid, provide a stronger missing-modality starting point than the recent
CFCompat correction line.

This benchmark performs **no new Test inference**.  It reuses prediction files
that were already frozen by historical experiments.

## ADPEP-All

ADPEP-All is directly applicable.  Stage 9B already froze label-free
predictions for `LAV`, `LA`, `LV`, and `L`.  The benchmark:

1. verifies `output_prediction_manifest.json` and the Test prediction SHA;
2. binds the label-free ADPEP rows to the frozen Stage 9A Test member rows;
3. recomputes all common metrics from the frozen predictions;
4. replays `adpep_metrics.csv` when that aggregate artifact exists;
5. compares ADPEP-All samplewise to the frozen Seed1113 Original CFCompat
   prediction, without creating a new NTR baseline definition.

## Historical Hybrid

The historical `MAE=0.6995` Hybrid was not originally a missing-modality model.
Its V7.1 implementation used five CFCompatKD best-validation checkpoints as raw
DLF teachers and selected, on validation only, a region-simplex committee plus
conservative shrinkage.

A missing-modality extension is reported only if all of the following frozen
facts are verified locally:

- the five teacher seeds are exactly `1111..1115`;
- the base teacher index is exactly `3` (Seed1114);
- the selected committee is `region_simplex`;
- the selected Hybrid beta is exactly `0.5`;
- the calibrated student alpha is exactly `0.0`, so the deployed historical
  Hybrid is algebraically `0.5 * region_committee + 0.5 * anchor`;
- the historical Hybrid Test prediction file reproduces the archived metrics
  (`MAE 0.6995`, `Corr 0.7942`, `Acc2 0.8491`, `F1 0.8484`, `Acc7 0.4781`,
  `Acc5 0.5394`) within the historical `5e-4` replay tolerance;
- rebuilding its LAV prediction from the Stage 9A five-member predictions,
  frozen region weights, region temperature, and beta reproduces the historical
  sample prediction within `2e-5` maximum absolute difference.

Only after these checks pass is the **same frozen formula** applied to the
already-frozen Stage 9A `LA/LV/L` member predictions.  This result is named
`hybrid_v71_missing_extension`; it is not relabeled as the original historical
Hybrid.

If any check fails, Hybrid is reported as unavailable rather than assigning it
an artificial missing-modality score.

## Baseline binding

Stage 9A Seed1113 is first required to replay the just-completed exploratory
Original-CFCompat Test aggregate (`TestJ`, MissingMacro MAE, and all four mode
MAEs) within `2e-6`.  This prevents comparing ADPEP/Hybrid against a different
historical Original checkpoint.

## Outputs

Only aggregate outputs are newly written:

- `adpep_hybrid_missing_comparison.csv`
- `adpep_hybrid_missing_summary.json`

No new sample-level Test output is written.

## Windows command

```powershell
.\scripts\run_windows_adpep_hybrid_missing_benchmark_v1.ps1
```

Use `-Overwrite` only to replace an existing aggregate benchmark output.  It
does not rerun Test inference; it only rereads the same frozen prediction files.
