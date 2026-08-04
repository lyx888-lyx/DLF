# CFCompatKD seed-mechanism audit v1

## Purpose

Explain why the selected SAM setting improves MOSI Seed 1111 but degrades Seed
1114 before defining another training method. This branch is analysis-only.

## Frozen evidence

- Source branch: `feature/cfcompat-sam-valid-screen-v1`.
- Source artifacts: the independently audited two-seed SAM Valid screen.
- Seeds: `1111` and `1114`.
- Compared checkpoints:
  - original `adam_replay`;
  - the SAM rho selected by mean two-seed Valid J.
- Official Test is forbidden.
- No checkpoint selection, model training, optimizer step, or parameter update is
  performed.

## Analysis 1: per-sample prediction effect

The audit binds the existing official Valid predictions by
`(Seed, sample_index, sample_id, label)` and computes, for each LAV/LA/LV/L view:

`delta_abs_error = SAM absolute error - baseline absolute error`.

Negative values mean SAM improved the sample. It also computes the per-sample J
proxy using the same official J weights, label/intensity bins, source-video
summaries, and cross-seed agreement.

A frozen Teacher LAV prediction and frozen Stage-1 evaluator predictions are
computed on Valid for descriptive context. The evaluator full-versus-missing
rank is named `valid_compat_proxy`; it is not the train-time compatibility gate
and may not be reported as such.

## Analysis 2: prediction shrinkage and calibration

For every seed, view, and variant the audit reports prediction mean/std,
mean absolute magnitude, MAE, label-to-prediction slope, intercept, and
correlation. SAM-to-baseline ratios test whether SAM mainly contracts prediction
scale rather than learning a shared improvement.

## Analysis 3: train-only objective-gradient conflict

A deterministic 64-sample train probe is selected with:

- exactly 64 unique samples;
- at most two samples per source video;
- deterministic hash ordering;
- approximate negative/near-zero/positive balance.

No train sample is used for fitting. At each frozen checkpoint, the audit
computes exact gradient cosine statistics for:

- `full` versus `missing`;
- `full` versus `KD`;
- `missing` versus `KD`;

for each missing mode LA/LV/L and parameter group:

- missing-token and mask-adapter path;
- final fusion head;
- non-BERT multimodal body;
- top BERT encoder layer.

The model is in training mode so the measurement reflects the trained objective,
but each comparison uses fixed RNG. No `backward`, optimizer, scheduler, or
parameter update is used. Model-state hashes before and after the audit must be
identical.

## Outputs and interpretation

The audit may support one or more descriptive mechanisms:

- cross-seed sample effects are shared;
- prediction shrinkage is supported;
- SAM changes gradient conflict consistently.

These flags do not authorize a new method. They only determine what mechanism
should be reviewed next. Any later training method requires a separate frozen
protocol.

## Hard stops

- No official Test construction or traversal.
- No new seed, rho, checkpoint, or best-case selection.
- No PCGrad, reweighting, calibration, or other training during this audit.
- No claim that the Valid compatibility proxy is the original train gate.
- No new method is authorized automatically by this analysis.
