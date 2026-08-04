# CFCompatKD + Video-Aware Batch Sampling v1

## Goal

Test whether source-video-aware training batches improve the generalization of
CFCompatKD without adding a loss, model component, or inference-time operation.
This is the sampler-only mechanism discovered in the Video-VREx experiment.
V-REx is not part of this method.

## Frozen base

- Base branch: `experiment/cfcompat-distillation-evidence-v1`.
- Fast-screen dataset: official CMU-MOSI train/valid/test splits.
- Original Student initialization, frozen Teacher, validation-best Stage-1
  evaluator, and train-only compatibility cache are reused without change.
- The complete objective remains exactly:
  `L_full + L_missing + L_CFCompatKD`.
- Validation selection remains:
  `J = 0.5 * MAE_LAV + 0.5 * mean(MAE_LA, MAE_LV, MAE_L)`.
- Test is not constructed during training or checkpoint selection.

## Sampler

The source video ID is parsed from each CMU segment ID. Each epoch:

- every training sample appears exactly once;
- no sample is duplicated or dropped;
- sampling uses no replacement and no oversampling;
- batch size is unchanged;
- the sampler takes up to four shuffled segments from a selected video;
- multiple source videos remain present in each batch;
- the sampler sequence is deterministic for the training seed.

`samples_per_video=4` is frozen from the discovery run. No alternative grouping
size, batch size, probability, or weighting is searched.

## Independent confirmation order

Seed 1114 is run first because seed 1111 produced the discovery observation.
Each formal seed performs:

1. an original shuffled-loader Stage-3 replay;
2. a sampler-only trajectory with the original CFCompatKD objective;
3. validation-only checkpoint selection;
4. one frozen Test traversal only after the validation gate passes;
5. an independent artifact audit.

When seed 1111 is later run, its validation result is also checked against the
previous Video-VREx sampler-control artifact when that artifact is available.

## Gates

The sampler candidate must pass all validation checks:

- Valid J gain at least `0.005`;
- Valid LAV MAE does not degrade;
- Valid MissingMacro MAE does not degrade;
- at least two epochs improve Valid J by at least `0.003`.

Only then is Test traversed once. Test passes when either:

- Test J gain is at least `0.005`; or
- Test J gain is positive and no LAV/LA/LV/L MAE degrades by more than `0.002`.

## Hard stops

- No V-REx, SAM, Mixup, router, expert, new loss, or test-time adaptation.
- No search over `samples_per_video`, batch size, or sampling policy.
- Seed 1114 Valid or Test failure stops the method family.
- Test results may not be used to change the sampler.
- Only two passing MOSI seeds authorize a MOSEI transfer.
