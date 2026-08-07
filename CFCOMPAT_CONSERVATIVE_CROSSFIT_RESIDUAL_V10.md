# CFCompatKD v10 — Conservative Cross-Fit Residual Selection

## Mechanism question

v9 showed that five video-grouped residual banks agreed on correction direction for almost every Valid event, yet the resulting correction magnitude was much larger than v8 and substantially damaged Teacher-beneficial samples.  This suggests a common-mode correction-strength problem rather than a simple fold-disagreement problem.

v10 asks one narrow question:

> Can we keep most of the Train-video-holdout benefit while selecting a materially earlier, smaller-residual checkpoint for each fold, thereby retaining v8-style safety without giving up all of v9's nonbeneficial repair?

## Frozen design

Everything below is unchanged from v9:

- Seed1113 only for this mechanism trajectory.
- Five deterministic video-grouped Train folds.
- Identical v8 residual-head architecture in every fold.
- Entire historical S0 path frozen and forced to eval mode.
- v4 DISTILL / PRESERVE / ABSTAIN objective unchanged.
- Same residual initialization across folds.
- Same 4-of-5 same-sign consensus and median correction at inference.
- No learned inference gate.
- No fitted Train/Valid calibration statistics.
- Official Test is never constructed or accessed.

## The only mechanism change

For each fold:

1. Train exactly as v9 and evaluate every epoch only on that fold's held-out Train videos.
2. Let `J_best` be the minimum Train-video-holdout J reached anywhere in that trajectory.
3. Define the frozen near-optimal band as:

   `J_epoch <= 1.01 * J_best`

4. Select the **earliest epoch** in that band.
5. Freeze that residual bank.

The 1% tolerance is fixed before the v10 run and does not use Official Valid.

## Why earliest-near-optimal instead of absolute best

v9 fold checkpoints were often very late (14–50 epochs), while v8's Official-Valid optimum occurred much earlier and had a far smaller residual magnitude.  The hypothesis is that Train-video holdouts reward continued correction long after correction strength becomes unsafe across the Train→Valid split.

The conservative selector therefore treats tiny late Train-holdout improvements as insufficient evidence to justify a much larger correction.

## Audit diagnostics

For every fold v10 writes both the v9-style absolute-best checkpoint and the conservative checkpoint, plus:

- absolute-best epoch and J,
- conservative epoch and J,
- relative J degradation,
- epoch reduction,
- absolute-best holdout mean/p95 residual magnitude,
- conservative holdout mean/p95 residual magnitude,
- conservative / best residual-magnitude ratio.

This directly tests whether the selector is buying a meaningful reduction in correction strength for only a small Train-holdout cost.

## Official Valid protocol

Official Valid is not used for:

- fold training,
- fold checkpoint selection,
- near-optimal tolerance selection,
- residual shrinkage calibration.

Only after all five conservative fold banks are frozen is the final consensus model evaluated on Official Valid.

## Frozen mechanism signal thresholds

The success gate is deliberately identical to v9 so that checkpoint selection is the only mechanism change:

- Valid J degradation versus frozen v8: <= 0.002.
- Teacher-beneficial NTR degradation versus frozen v8: <= 0.03.
- Teacher-nonbeneficial NTR reduction versus frozen v8: >= 0.05.
- Overall NTR degradation versus frozen v4: <= 0.01.

## Key outputs

`result/missing_baseline/cfcompat_conservative_crossfit_residual_v10/mosi/valid_screen/seed1113_dev/`

Important files:

- `conservative_crossfit_v10_valid_screen_summary.json`
- `conservative_crossfit_v10_transfer_summary.csv`
- `conservative_crossfit_v10_fold_manifest.csv`
- `conservative_crossfit_v10_fold_epoch_metrics.csv`
- `conservative_crossfit_v10_valid_consensus_events.csv`

## Interpretation

A positive mechanism signal would require the conservative checkpoints to reduce residual magnitude materially, recover much of v8's beneficial-sample safety, and still preserve meaningful nonbeneficial repair.

A negative result would imply that the cross-split problem is not merely late-epoch overcorrection; the next hypothesis would need to model sample-specific correction magnitude uncertainty rather than only selecting a globally earlier residual state.
