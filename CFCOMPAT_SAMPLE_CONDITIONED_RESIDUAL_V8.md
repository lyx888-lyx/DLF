# CFCompatKD v8 — Frozen-S0 Sample-Conditioned Residual

## Question

v7 showed a sharp decomposition on Seed1113 Valid:

- hard isolation strongly protected Teacher-beneficial events;
- the 925-parameter mode-only adapter could not adequately repair the
  Teacher-nonbeneficial complement.

v8 tests whether more **isolated, sample-conditioned correction capacity** can
recover the complementary cases without reopening shared-S0 drift.

## Architecture

The complete historical S0 path is immutable:

- DLF backbone: frozen;
- missing audio token: frozen;
- missing vision token: frozen;
- historical mask adapter: frozen;
- all S0 buffers/module state: frozen and forced to eval mode.

Only three new residual heads are trainable: one each for `LA`, `LV`, and `L`.

For each missing sample, v8 builds a detached frozen feature vector from:

- `c_l_sim`, `c_v_sim`, `c_a_sim`;
- frozen `logits_c`;
- frozen L/V/A heterogeneous logits;
- frozen S0 final prediction.

The feature vector uses parameter-free per-sample LayerNorm.  No Train-fitted
mean/variance, label, sample ID, or split-level calibration statistic is an
inference input.

Each mode has an independent MLP:

`frozen sample feature -> Linear -> GELU -> Linear -> bounded scalar residual`

The final layer is initialized to zero, so epoch 0 is exactly S0.  The residual
is bounded by `tanh` to `[-1, 1]`.  LAV receives exactly zero residual.

## Training

The v4 `DISTILL / PRESERVE / ABSTAIN` objective is reused unchanged.  Only the
new residual heads can receive gradients.

One development trajectory is trained:

- dataset: MOSI
- seed: 1113
- checkpoint selection: minimum official Valid J
- Test construction/access: forbidden

Per-epoch sample-level Valid predictions are saved again.

## Frozen mechanism checks

Before the run, v8 fixes the following checks:

1. Valid J may degrade by at most `0.002` versus v7.
2. Teacher-beneficial NTR may degrade by at most `0.03` versus v7.
3. Teacher-nonbeneficial NTR must improve by at least `0.10` versus v7.
4. Overall NTR may be at most `0.01` worse than frozen v4.

These are mechanism-signal checks, not a hyperparameter grid.

## Why this may generalize better across splits

v8 deliberately removes several split-sensitive paths:

- no label-trained inference gate;
- no shared-backbone update;
- no Train-fitted feature normalization;
- no unbounded correction;
- correction depends on the current sample's frozen representation and missing
  mode rather than a global Train calibration rule.

This design targets cross-split robustness.  It does **not** establish that the
official Test distribution will improve; Test remains locked under the current
protocol.

## Main outputs

`result/missing_baseline/cfcompat_sample_conditioned_residual_v8/mosi/valid_screen/seed1113_dev/`

Important files:

- `sample_residual_v8_valid_screen_summary.json`
- `sample_residual_v8_transfer_summary.csv`
- `sample_residual_v8_valid_epoch_transfer_summary.csv`
- `sample_residual_v8_valid_epoch_event_trajectory.csv`
- `sample_residual_v8_valid_failure_onset.csv`
- `sample_residual_v8_valid_clip_failure_epoch_summary.csv`
- `sample_residual_v8_residual_valid_events.csv`
- `sample_residual_v8_trainable_parameter_delta.csv`
