# FixedBlend-DP57-v1 protocol

## Goal

Recover the strong ordinal classification decisions associated with the ADPEP
anchor while retaining as much as possible of the continuous regression quality
of the frozen Raw5/v13 50/50 FixedBlend.

This method does **not** average ADPEP and FixedBlend predictions.  Instead, it
uses the ADPEP-57 decision geometry:

1. choose one anchor seed using validation J only;
2. form the already-frozen FixedBlend prediction
   `0.5 * Raw5-PE5 + 0.5 * v13`;
3. for each sample and each mode, compute the maximal float32 interval in which
   the anchor's Acc7 and Acc5 evaluator decisions remain unchanged;
4. output the point in that interval closest to the FixedBlend prediction.

The projection is label-free.  Labels are used only after prediction freezing to
compute aggregate metrics.

## What is guaranteed

For every sample and mode, the projected prediction has exactly the same Acc7
and Acc5 decision as the anchor.  Therefore aggregate Acc7 and Acc5 are exactly
inherited from the anchor, subject only to successful input/sample binding.

The projection is the minimum-distance modification of FixedBlend subject to
those two decision constraints.  MAE, Corr, Acc2 and F1 are therefore retained
as much as this constraint geometry permits, but they are not mathematically
guaranteed to remain identical to FixedBlend.

## MOSI development rule

MOSI Test is blocked for this new method.  MOSI is used only for a Valid audit:

```powershell
.\scripts\run_windows_fixedblend_dp57_valid_v1.ps1
```

The audit reports:

- exact Acc7/Acc5 inheritance;
- FixedBlend vs DP57 J/MAE/Corr/Acc2/F1/Acc7/Acc5;
- projection rate and mean absolute perturbation;
- the rate at which enforcing Acc7/Acc5 changes the FixedBlend Acc2 decision.

No projected sample-level prediction file is persisted.

## MOSEI protocol

The projection implementation already supports MOSEI evaluator decisions.
MOSEI requires five frozen Raw5 member prediction files and one frozen v13
prediction file for each evaluated split.

### Valid

Run Valid first.  The anchor seed is selected by the five fixed seeds' Valid J:

```powershell
.\scripts\run_windows_fixedblend_dp57_mosei_v1.ps1 `
  -Split valid `
  -V13Predictions <MOSEI_V13_VALID_PREDICTION_CSV>
```

Record the printed `Anchor seed`.

### Test

Test may only use the anchor frozen above:

```powershell
.\scripts\run_windows_fixedblend_dp57_mosei_v1.ps1 `
  -Split test `
  -AnchorSeed <VALID_FROZEN_SEED> `
  -V13Predictions <MOSEI_V13_TEST_PREDICTION_CSV>
```

The Test evaluator is a separate program and cannot select an anchor from Test.
No weight search, anchor search, calibration, or sample-level projected Test
artifact is permitted.

## Relationship to historical ADPEP-All

Historical ADPEP-All preserves the anchor's Acc7, Acc5 and Acc2 decisions.
FixedBlend-DP57 intentionally preserves only Acc7 and Acc5.  This is deliberate:
forcing the anchor's Acc2 decision would also force its binary-classification
profile, which can discard the stronger FixedBlend Acc2/F1 behavior.  DP57 is
therefore the correct constraint set for the present goal: inherit the strong
Acc7/Acc5 profile while keeping the newer continuous/binary behavior whenever
compatible.
