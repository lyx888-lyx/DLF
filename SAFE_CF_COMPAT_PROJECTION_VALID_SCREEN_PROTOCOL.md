# Safe-CFCompatKD v1 held-out-seed Valid screen

## Purpose

This screen tests one mechanism derived from the frozen MOSI mechanism audit:
original CFCompatKD can help high-error samples when the complete-modality
Teacher points toward the label, but it can damage easy samples when Teacher
movement is wrong-direction or overshoots the label.

The screen does not tune a penalty magnitude.  It changes only the training KD
target.  All inference-time architecture and parameters remain unchanged.

## Frozen safe target

For train sample `i` and sampled missing view `m`:

- `b_i^m`: prediction from the frozen validation-best DLF-ModDrop evaluator;
- `t_i`: prediction from the frozen complete-LAV Teacher;
- `y_i`: train regression label.

The safe Teacher target is

```text
safe_t_i^m = clip(t_i, min(b_i^m, y_i), max(b_i^m, y_i)).
```

Consequences:

- a Teacher already inside the baseline-to-label interval is unchanged;
- a wrong-direction Teacher is projected to the frozen baseline endpoint;
- a correct-direction Teacher that crosses the label is clipped to the label;
- a zero-width baseline-to-label interval cannot be disturbed by KD.

The operation is train-only.  Validation and later inference use the Student
alone and do not require labels, Teacher, baseline evaluator, or projection.

## Fixed runs

Each held-out seed runs exactly three trajectories from the same clean DLF
checkpoint, with the original Adam optimizer, `update_epochs=10`, missing-mask
sequence, early stopping, and minimum official-Valid-J checkpoint selection.

1. `cfcompat_replay`: original CFCompatKD target and original compatibility gate.
2. `safe_uniform`: safe target with unit gate; compatibility is removed.
3. `safe_cfcompat`: safe target with the original compatibility gate.

`safe_cfcompat` is the main candidate.  `safe_uniform` is an ablation and is not
selected against `safe_cfcompat` by Valid score.

## Held-out seeds

The audit that motivated the mechanism used Seeds 1111 and 1114.  Formal
screening therefore fixes previously unused mechanism-validation seeds:

```text
1112, 1113, 1115
```

No seed subset, replacement, or post-hoc exclusion is allowed.

## Data and Test lock

- Dataset: CMU-MOSI.
- Training split: official train.
- Decision split: official Valid only.
- Official Test loader construction and traversal: forbidden.
- MOSEI: forbidden unless the main candidate passes every fixed gate.

## Primary promotion checks

Relative to the exact original-CFCompatKD replay, `safe_cfcompat` must satisfy
all of the following:

1. positive Valid-J gain on all three held-out seeds;
2. mean Valid-J gain at least `0.005`;
3. no seed and no LAV/LA/LV/L mode degradation above `0.002`;
4. at least two epochs per seed with Valid-J gain at least `0.003`.

## Failure-repair mechanism checks

Difficulty quartiles are defined from the frozen DLF-ModDrop Valid absolute
error independently within each seed and view.  The candidate must satisfy all
checks on every held-out seed:

1. when original CFCompatKD damages Q1-easy samples, remove at least 50% of that
   damage; when it does not, remain within `0.002` of its Q1 gain;
2. when original CFCompatKD helps Q4-hard samples, retain at least 80% of that
   gain; when it does not, remain within `0.002`;
3. strictly reduce the harmful-imitation rate;
4. keep the `Teacher better and direction correct` gain within `0.002` of
   original CFCompatKD.

Main promotion requires every primary and mechanism check.  The only main-pass
verdict is:

```text
PROMOTE_SAFE_CFCompatKD_TO_MOSEI_SINGLE_SEED_VALID_SCREEN
```

If only `safe_uniform` passes, the result is recorded as an ablation result and
does not validate the CFCompat extension.  Otherwise the method is stopped.

## Forbidden adaptations

After observing Valid results, do not:

- add a damage penalty;
- tune KD weight, temperature, interval width, or projection softness;
- choose seeds, videos, labels, or Teacher subsets;
- change Q1/Q4 definitions or thresholds;
- reverse or reshape compatibility;
- access MOSI Test;
- start MOSEI after a failed main gate.
