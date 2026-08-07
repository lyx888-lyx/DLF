# CFCompatKD v9 — Cross-Fit Residual Consensus

## Question

v8 showed that the sample-conditioned residual has enough capacity to repair
Teacher-nonbeneficial cases, but stronger correction simultaneously destroys
Teacher-beneficial cases. v9 asks whether the harmful corrections are less
stable across changes in the Train distribution than the useful corrections.

## Architecture

The complete historical Student S0 remains immutable. v9 reuses the exact v8
residual architecture:

- three mode-specific residual heads: LA / LV / L;
- detached frozen S0 features;
- hidden dimension 64;
- bounded scalar residual with max absolute value 1.0;
- no correction on LAV.

No additional representation capacity is introduced.

## Cross-fit training

Train clips are grouped by MOSI video id parsed from `sample_id`. Complete videos
are assigned to five deterministic approximately balanced folds. For each fold:

1. the residual head bank is initialized identically;
2. the held-out video fold is excluded from gradient updates;
3. the bank is trained on the other four folds using the unchanged v4
   DISTILL/PRESERVE/ABSTAIN objective;
4. checkpoint selection and early stopping use only the held-out **Train video**
   objective J;
5. official Valid is not consulted.

This yields five residual banks whose differences are driven primarily by which
Train videos were excluded.

## Inference consensus

For a missing-modality sample, the five banks produce corrections
`delta_1 ... delta_5`.

A correction is applied only if at least **4 of 5** banks agree on its sign.
When that condition holds, v9 uses the median correction. Otherwise v9 applies
zero correction and falls back exactly to S0.

This rule is fixed before the Valid run. It uses:

- no labels;
- no learned gate;
- no Train/Valid calibration statistics;
- no fitted normalization statistics.

## Why this is a distribution-shift experiment

v5 showed Train-to-Valid conditional relation shift for learned usefulness
geometry. v8 showed that a single residual model can learn useful corrections
but cannot reliably decide where they remain safe.

v9 therefore tests a different quantity: whether a correction is stable to
removing whole Train videos. Agreement is used as a robustness signal rather
than a classifier prediction of Teacher usefulness.

A positive v9 result would support the hypothesis that useful corrections are
more training-subset-stable than harmful split-specific corrections. It would
not by itself guarantee Official Test improvement.

## Protocol

- Dataset: MOSI.
- Development seed: 1113.
- Cross-fit folds: 5, grouped by video id.
- Residual initialization: identical across folds.
- Fold checkpoint selector: minimum held-out Train-video J.
- Official Valid: used only after all five fold models are trained and frozen.
- Official Test: never constructed or accessed.
- No hyperparameter grid.
- No repeat Test.

## Frozen mechanism signal

Before the run:

- Valid J may degrade by at most 0.002 vs frozen v8.
- Teacher-beneficial NTR may degrade by at most 3 percentage points vs v8.
- Teacher-nonbeneficial NTR must improve by at least 5 percentage points vs v8.
- Overall NTR may be at most 1 percentage point worse than frozen v4.

These checks are mechanism-screen criteria, not a tuning grid.

## Main outputs

`result/missing_baseline/cfcompat_crossfit_residual_consensus_v9/mosi/valid_screen/seed1113_dev/`

Key files:

- `crossfit_residual_v9_valid_screen_summary.json`
- `crossfit_residual_v9_candidate_grid.csv`
- `crossfit_residual_v9_transfer_summary.csv`
- `crossfit_residual_v9_fold_assignment.csv`
- `crossfit_residual_v9_fold_manifest.csv`
- `crossfit_residual_v9_fold_epoch_metrics.csv`
- `crossfit_residual_v9_fold_train_decisions.csv`
- `crossfit_residual_v9_valid_consensus_events.csv`
- `crossfit_residual_v9_candidate_raw_valid_events.csv`

The consensus event table contains all five fold corrections, sign agreement,
fold dispersion, the median correction, and whether consensus was applied.
