# DLF MOSI long-tail and semantic-risk coupling audit v1.1

## Purpose

Determine whether the established MOSI seven-bin label imbalance is actually
coupled to larger prediction error and higher-cost semantic mistakes before
training any long-tail weight or semantic-risk loss.

This audit is separate from the rejected hard role-specialization hypothesis.
It does not require shared representations to specialize in polarity or
specific representations to specialize in intensity.

## Frozen source

- Source branch: `analysis/dlf-role-specialization-audit-v1`.
- Source artifacts: the independently passed v2 role-specialization audit.
- Source predictions: frozen CFCompatKD `adam_replay` checkpoints for seeds
  `1111` and `1114`.
- Frequency source: official MOSI train labels only.
- Error source: official MOSI Valid predictions only.
- Views: `LAV`, `LA`, `LV`, and `L`.
- Official Test is forbidden.
- No checkpoint, model forward, training, optimizer, backward pass, or parameter
  update occurs in this branch.

## Frequency groups

The seven fixed sentiment bins are `-3,-2,-1,0,1,2,3` with the boundaries
already audited by the role-specialization study.

Train frequency alone determines the groups:

- tail: the three lowest-frequency non-empty bins;
- head: the three highest-frequency bins;
- middle: the remaining bin.

Ties use stable ascending sentiment-bin order. Valid error is never used to
choose tail or head bins.

## Per-sample risk events

For each frozen Valid prediction the audit records:

- absolute error;
- direct positive-negative polarity flip;
- non-neutral prediction collapsed to neutral;
- neutral prediction escaping to non-neutral;
- catastrophic absolute error of at least `2.0`;
- severe opposite-polarity error for labels with `|y|>1.5`.

The primary high-cost event is fixed as:

`direct positive-negative flip OR absolute error >= 2.0`.

The other event rates remain descriptive and are not silently merged into the
primary decision.

## Run-level evidence

For each of two seeds and four views, the audit computes:

- train-count versus Valid bin-MAE Spearman correlation;
- train-count versus Valid high-cost-rate Spearman correlation;
- tail-minus-head macro-bin MAE;
- tail-minus-head sample-weighted MAE;
- tail-minus-head high-cost-event rate;
- tail-minus-head direct-flip rate;
- tail-minus-head non-neutral-to-neutral rate;
- whether the worst-MAE bin belongs to the train-defined tail.

Negative count-risk correlation and positive tail-head gaps support coupling.

## Joint video bootstrap

The official MOSI Valid set contains few source videos. Therefore uncertainty is
estimated by a deterministic joint video-cluster bootstrap:

- resample source videos with replacement;
- use the same resampled videos for all seeds and views;
- preserve all segments from a selected video;
- require every accepted draw to contain all three fixed Tail bins and all three
  fixed Head bins, so the six-bin macro statistic has the same definition in
  every replicate;
- reject a draw that omits any required Tail/Head bin and continue sampling;
- collect exactly `2000` valid replicates with seed `20260804`;
- permit at most `100 × requested_replicates` total attempts, with a minimum
  ceiling of `10000` attempts;
- record accepted draw attempt numbers, cumulative rejected draws, rejection
  fraction, and percentile 95% intervals.

Rejected draws are not assigned zero and are not retained as `NaN`. If the fixed
number of valid replicates cannot be collected within the attempt ceiling, the
audit stops and reports that Valid video support is too sparse for this frozen
bootstrap definition.

Segments are not treated as independent bootstrap units.

## Separate frozen gates

### Tail-MAE coupling

All conditions are required:

- both LAV seeds have positive tail-head macro-MAE gaps;
- at least 75% of all eight runs have positive gaps;
- at least 75% have negative count-versus-MAE correlation;
- mean tail-head macro-MAE gap is at least `0.05`;
- mean count-versus-MAE Spearman is at most `-0.25`;
- joint video-bootstrap 95% lower bound is positive.

### Semantic-risk coupling

All conditions are required:

- both LAV seeds have positive tail-head high-cost-rate gaps;
- at least 75% of all eight runs have positive gaps;
- at least 75% have negative count-versus-high-cost correlation;
- mean high-cost-rate gap is at least `0.01`;
- mean count-versus-high-cost Spearman is at most `-0.20`;
- joint video-bootstrap 95% lower bound is positive.

## Verdicts

- `PROMOTE_SEPARATE_FINAL_TAIL_AND_RISK_SCREENS`: both mechanisms pass; they
  must still be trained separately before any combination.
- `PROMOTE_FINAL_TAIL_WEIGHT_SCREEN_ONLY`: only MAE coupling passes.
- `PROMOTE_FINAL_SEMANTIC_RISK_SCREEN_ONLY`: only high-cost-risk coupling passes.
- `PARTIAL_TAIL_RISK_COUPLING_DO_NOT_TRAIN`: an average positive signal exists,
  but the stability gate is incomplete.
- `STOP_TAIL_RISK_COUPLING_NOT_SUPPORTED`: imbalance exists without consistent
  error coupling.
- `STOP_LONG_TAIL_PREMISE_NOT_SUPPORTED`: the source train distribution is not
  long-tailed under the existing frozen criterion.

No verdict in this branch authorizes a combined loss. A promoted mechanism
requires a separate training branch with its own frozen Valid gate.

## Hard stops

- No official Test construction or traversal.
- No new checkpoint or seed selection.
- No manual tail-bin definition after observing Valid results.
- No manual class weights.
- No training of tail weighting, focal loss, risk loss, calibration, or role
  specialization.
- No claim that label imbalance alone implies a beneficial weighted objective.
