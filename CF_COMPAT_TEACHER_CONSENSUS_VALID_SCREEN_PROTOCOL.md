# CFCompatKD multi-teacher consensus Valid screen v1

## Purpose

Test one final extension that is native to CFCompatKD rather than an external
optimizer or data-augmentation plugin. Original CFCompatKD estimates whether a
complete-modality Teacher is compatible with a missing-modality counterfactual.
This screen additionally asks whether the complete-modality teaching signal is
stable across independently initialized Teachers.

The two quantities have distinct meanings:

- counterfactual compatibility: whether complete-modality knowledge applies to
  the current missing-modality view;
- Teacher consensus: whether that complete-modality target is stable across
  training seeds.

## Frozen scope

- Dataset: MOSI only.
- Formal Student seeds: `1111` and `1114`.
- Complete-modality Teacher seeds: exactly `1111,1112,1113,1114,1115`.
- Decision split: official Valid only.
- Official Test is forbidden.
- Base optimizer: original Adam.
- Original `update_epochs=10` gradient-accumulation window.
- Original train missing-mask sequence and early-stop rule.
- Original complete-view loss, missing-view loss, compatibility caches and
  validation objective.
- No Teacher count, subset, temperature, variance transform, gate exponent or
  coefficient search.
- No additional inference parameters.

## Train-only Teacher cache

Before Student training, each of the five clean DLF checkpoints is loaded
sequentially and evaluated in LAV mode on a deterministic, non-shuffled official
train loader. Only one Teacher is resident at a time.

For train sample `x`, let the five predictions be `T_k(x)`. The frozen cache
stores:

`mu_T(x) = mean_k T_k(x)`

`u_T(x) = mean_k (T_k(x) - mu_T(x))^2`

The fixed uncertainty scale is the median population variance over all official
train samples:

`s_u = median_train u_T(x)`

The consensus weight is:

`r_T(x) = 1 / (1 + u_T(x) / s_u)`

The median must be finite and strictly positive. Cache construction preserves all
training RNG states and records every Teacher checkpoint SHA256. No Valid label
or prediction is used to construct the cache.

## Three locked runs

Every formal Student seed runs the same three trajectories from its original
clean checkpoint.

### 1. `single_teacher_replay`

Exact original CFCompatKD replay:

- target: the matching-seed clean Teacher LAV prediction;
- gate: original counterfactual compatibility;
- objective: `L_full + L_missing + L_CFCompatKD`.

The validation-best epoch and all four Valid MAEs must reproduce the historical
Stage-3 result within `1e-4`.

### 2. `ensemble_mean`

Mandatory ablation:

- target: `mu_T(x)`;
- gate: original counterfactual compatibility;
- all other training details unchanged.

This isolates changing the Teacher target from uncertainty calibration.

### 3. `ensemble_consensus`

Primary extension:

- target: `mu_T(x)`;
- gate: `counterfactual_compatibility * r_T(x)`;
- all other training details unchanged.

Because gated SmoothL1 is normalized by total gate mass, the consensus term
changes the relative sample weighting rather than applying an arbitrary global
loss multiplier.

## No candidate selection

`ensemble_mean` and `ensemble_consensus` each have an independent frozen gate.
The lower mean Valid J is not used to select a candidate. The consensus run is
the primary method; the mean run is an ablation that can only be promoted under
its own complete gate.

## Dual-seed Valid gate

For each candidate independently, all conditions are required:

- both Student seeds have positive Valid-J gain over the historical CFCompatKD
  baseline;
- mean two-seed Valid-J gain is at least `0.005`;
- no seed and no LAV/LA/LV/L mode degrades by more than `0.002` MAE;
- LAV is not materially degraded for either seed;
- missing-mode macro MAE is not materially degraded for either seed;
- each seed has at least two epochs whose Valid-J gain is at least `0.003`.

## Verdicts

- `PROMOTE_TEACHER_CONSENSUS_TO_MOSEI_SINGLE_SEED_VALID_SCREEN`: the primary
  consensus-calibrated run passes.
- `PROMOTE_ENSEMBLE_MEAN_ABLATION_TO_MOSEI_SINGLE_SEED_VALID_SCREEN`: consensus
  fails but the mean-target ablation independently passes.
- `STOP_TEACHER_CONSENSUS_DUAL_SEED_VALID_FAILED`: neither candidate passes.

A promotion authorizes only a separately frozen MOSEI single-seed Valid screen.
It does not authorize MOSI Test, MOSEI Test, Teacher-subset tuning or combining
this method with SAM, long-tail weighting or semantic-risk losses.

## Outputs

The screen writes:

- train-only five-Teacher prediction cache and configuration;
- six-run Valid grid;
- all epoch-level Valid metrics;
- all six validation prediction tables;
- source/checkpoint manifest;
- machine-readable summary and report;
- independent audit output.

## Hard stops

- No official Test construction or traversal.
- No Teacher subset or seed selection.
- No uncertainty hyperparameter expansion.
- No replacement of the original compatibility cache.
- No live five-Teacher ensemble during Student training.
- No additional inference network or parameter.
- A failed dual-seed gate ends this direction.
