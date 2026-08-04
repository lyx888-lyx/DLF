# MOSI dataset limitations and CFCompatKD mechanism audit v1

## Purpose

This is a frozen explanatory audit, not a new method search. It asks three
questions:

1. Which properties of MOSI make results unstable, heterogeneous, or easy to
overinterpret?
2. Where do DLF-ModDrop errors and modality contributions differ most across
videos, labels, missing views, and random seeds?
3. Which error pattern is actually reduced by CFCompatKD relative to the
validation-best DLF-ModDrop baseline?

## Frozen scope

- Dataset: MOSI.
- Splits constructed: official train and valid only.
- Official Test construction, traversal, labels, predictions, and per-sample
  analysis are forbidden.
- Student seeds: exactly `1111` and `1114`.
- Baseline: each seed's recorded validation-best DLF-ModDrop checkpoint.
- Method: each seed's recorded validation-best CFCompatKD checkpoint.
- Clean Teacher: each seed's original complete-modality DLF checkpoint.
- Counterfactual evaluator and compatibility cache: the frozen Stage-1
  validation-best DLF-ModDrop evaluator and train-only cache used by CFCompatKD.
- No training, optimizer, scheduler, backward pass, parameter update, checkpoint
  selection, hyperparameter search, or new method authorization.

Historical Stage-18 Valid control tables are read only when already present.
They are never required to make the main DLF-ModDrop versus CFCompatKD
comparison and are never reconstructed from Test.

## Dataset audit

For train and valid, the audit records:

- segment and source-video counts;
- effective source-video count and top-video concentration;
- seven-bin sentiment distribution, polarity, intensity, near-neutral fraction,
  and imbalance ratio;
- exact normalized-text duplicates and train-valid overlap;
- train-valid label mean, variance, and Jensen-Shannon distribution shift;
- per-video label range, adjacent-label correlation, adjacent polarity
  persistence, and segment count;
- text-tensor, audio, and vision finite values, active length, zero padding,
  all-zero samples, near-constant samples, and scalar feature magnitude.

The audit treats video as the uncertainty unit. Segment count is not interpreted
as the number of independent observations.

## Frozen validation predictions

For every seed, valid sample, and view `LAV`, `LA`, `LV`, and `L`, the audit
exports:

- DLF-ModDrop prediction;
- CFCompatKD prediction;
- clean LAV Teacher prediction;
- frozen evaluator LAV and missing-view predictions;
- evaluator counterfactual shift;
- a descriptive Valid compatibility proxy calibrated only against the
  corresponding train-cache shift distribution.

The Valid proxy is never claimed to be a training gate. It exists only to ask
whether the training principle generalizes descriptively to unseen samples.

## Mechanism variables

For each sample-view pair:

- baseline error: `|DLF-y|`;
- CFCompatKD error: `|CFCompatKD-y|`;
- gain: `|DLF-y|-|CFCompatKD-y|`;
- Teacher advantage: `|DLF-y|-|Teacher-y|`;
- Teacher-better indicator;
- Teacher direction-correct indicator;
- movement toward the Teacher;
- helpful imitation, harmful imitation, improvement without imitation, and
  double failure quadrants.

The primary explanatory hypothesis is not that compatibility predicts Teacher
correctness. It is that compatibility estimates whether complete-modality
knowledge remains applicable after a particular modality is removed, thereby
reducing indiscriminate harmful imitation.

## Difference and opportunity tables

Results are stratified by:

- modality view;
- seven-bin sentiment level;
- polarity and absolute intensity;
- compatibility decile;
- baseline-error quartile;
- Teacher-benefit/direction condition;
- source video.

The opportunity table is descriptive. A group is listed only when both seeds
contain at least 15 samples. Ranking prioritizes positive gain in both seeds,
then the worse-seed gain, remaining CFCompatKD error, Teacher advantage,
Teacher direction correctness, and compatibility. It does not authorize a new
loss or specialist.

## Video-cluster uncertainty

A deterministic joint source-video bootstrap uses the same resampled videos for
both seeds and all four views:

- 2000 finite accepted replicates;
- seed `20260804`;
- J gain recomputed from mode MAEs in every replicate;
- high-compatibility (`deciles 8-10`) and low-compatibility (`deciles 1-3`)
  gains recomputed in every replicate;
- invalid draws are rejected rather than filled with zero or NaN.

## Mechanism verdicts

- `CFCompat_IMPROVEMENT_MECHANISM_SUPPORTED`: both seeds improve in Valid J,
  the video-bootstrap lower bound is positive, recoverable Teacher conditions
  benefit more, and high compatibility is both more beneficial and not more
  harmful than low compatibility.
- `CFCompat_GAIN_REPRODUCED_MECHANISM_PARTIAL`: the two-seed gain is reproduced,
  but one or more explanatory checks are incomplete.
- `CFCompat_GAIN_NOT_REPRODUCED`: the frozen checkpoint comparison does not show
  positive Valid-J gain for both seeds.

A partial mechanism verdict does not invalidate the method's empirical result;
it limits the causal claim that can be made from this audit.

## Hard stops

- No Test access.
- No post-hoc compatibility threshold.
- No new router, specialist, tail weight, Teacher subset, or confidence formula.
- No selection of videos, label bins, or seeds for a new method.
- No claim that descriptive group differences are independent causal effects.
