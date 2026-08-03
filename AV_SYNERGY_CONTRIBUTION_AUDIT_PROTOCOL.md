# AV Synergy Contribution Audit v1

## Purpose

This stage decides whether the existing CFCompatKD counterfactual prediction lattice contains enough stable audio, vision, and audio-vision interaction signal to justify training a contribution-factorization distillation model.

It is a zero-training audit. It does not load a model, Teacher, evaluator, compatibility cache, optimizer, or checkpoint. It reads only the five Stage8 Online prediction CSVs selected by validation J.

## Frozen sources

For seeds 1111–1115 and splits Valid/Test:

`result/missing_baseline/cfcompat_stability_v1/mosi/seedSEED/online_SPLIT_predictions.csv`

Every source must contain exactly aligned `sample_index`, `sample_id`, `label`, `LAV_pred`, `LA_pred`, `LV_pred`, and `L_pred` columns. Source SHA-256 values are recorded and independently rechecked.

The formal decision uses Valid only. Test is a frozen diagnostic split and cannot change thresholds or the verdict.

## Counterfactual factorization

For each sample and seed:

- text base: `b = T_L`
- audio marginal effect: `a = T_LA - T_L`
- vision marginal effect: `v = T_LV - T_L`
- AV synergy effect: `s = T_LAV - T_LA - T_LV + T_L`
- no-synergy reconstruction: `T_no_syn = T_LA + T_LV - T_L`

Label-error gains are:

- audio gain: `|T_L-y| - |T_LA-y|`
- vision gain: `|T_L-y| - |T_LV-y|`
- synergy gain: `|T_no_syn-y| - |T_LAV-y|`
- total AV gain: `|T_L-y| - |T_LAV-y|`

Effect magnitude is never interpreted as usefulness without label-error gain.

## Evidence

The audit reports:

1. per-seed effect distributions and MAE gains;
2. five-seed equal-prediction ensemble effects and gains;
3. video-group bootstrap confidence intervals;
4. cross-seed effect rank and sign consistency;
5. high-magnitude sample coverage;
6. conditional regions such as high text error and A/V direction agreement;
7. video concentration and leave-one-video-out sensitivity;
8. a label-defined synergy oracle as diagnostic headroom only.

## Pre-registered gates

Synergy support requires on Valid:

- ensemble synergy MAE gain at least `0.003`;
- video-group bootstrap 95% CI lower bound above zero;
- positive mean synergy gain in at least 4/5 seeds;
- at least 5% of samples with mean absolute synergy effect at least `0.10`;
- mean cross-seed sign agreement at least `0.70` in that high-magnitude subset.

Marginal support requires audio or vision to satisfy:

- ensemble marginal MAE gain at least `0.002`;
- video-group bootstrap 95% CI lower bound above zero;
- positive mean gain in at least 4/5 seeds.

Verdicts:

- both pass: `SUPPORTED_BUILD_CONTRIBUTION_FACTORIZATION_DISTILLATION`;
- one passes: `PARTIAL_SIGNAL_EXPAND_ONLY_SUPPORTED_COMPONENT`;
- neither passes: `NOT_SUPPORTED_DO_NOT_TRAIN_CONTRIBUTION_FACTORIZATION`.

No result-directed threshold, seed, split, or component selection is permitted.
