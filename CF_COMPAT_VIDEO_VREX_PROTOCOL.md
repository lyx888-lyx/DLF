# CFCompatKD + Video-VREx v1

## Goal

Transfer the mature V-REx domain-generalization objective into missing-modality
multimodal sentiment regression. A source video is treated as one training
domain. The method targets cross-video risk instability without changing the DLF
inference architecture.

## Frozen base

- Base branch: `experiment/cfcompat-distillation-evidence-v1`.
- Dataset for the fast screen: official CMU-MOSI train/valid/test.
- Initial Student, frozen Teacher, Stage-1 evaluator and compatibility cache are
  the seed-matched locked CFCompatKD assets.
- Original objective remains unchanged:
  `L_full + L_missing + L_CFCompatKD`.
- Validation objective remains
  `J = 0.5 * MAE_LAV + 0.5 * mean(MAE_LA, MAE_LV, MAE_L)`.
- Test is never constructed during baseline replay, sampler control, lambda
  selection, or checkpoint selection.

## Video domain

The source video ID is parsed from every sample ID. The primary supported CMU
format is `VIDEO_ID$_$SEGMENT_ID`; bracketed numeric segment suffixes are also
accepted. Train, valid and test video sets must be pairwise disjoint.

## Batch construction

The locked Stage-3 trajectory is first replayed with the original shuffled
loader and must reproduce its validation result within `1e-4`.

V-REx runs use a deterministic video-aware batch sampler. Every training sample
appears exactly once per epoch and no sample is duplicated. The sampler tries to
place four samples from each selected video in a batch so that per-video risk is
estimable. A `lambda=0` sampler-only control is trained with the identical batch
sequence used by all positive-lambda candidates.

## V-REx objective

For sample `i`, define the supervised prediction risk

`r_i = |p_i^LAV - y_i| + |p_i^missing - y_i|`.

For every video represented by at least two samples in the batch,

`R_g = mean_{i in g}(r_i)`.

The penalty is the population variance across eligible video risks:

`L_VREx = Var_g(R_g)`.

The total objective is

`L = L_full + L_missing + L_CFCompatKD + lambda * L_VREx`.

The penalty uses true labels only. It never modifies the Teacher,
compatibility, gate, missing-mask sampling, or auxiliary DLF losses.

## Frozen lambda screen

Only the following positive coefficients are permitted:

- `0.01`
- `0.1`
- `1.0`

All candidates use the same seed, initialization, video-aware batch sequence,
missing-mask RNG sequence, Teacher and compatibility cache. The `lambda=0`
video-aware run is attribution control only and is not eligible for selection.
The positive lambda with the lowest validation J is selected; ties prefer the
smaller lambda.

## Promotion gates

The selected positive-lambda run must first pass the validation-only gate:

- validation J gain over locked CFCompatKD at least `0.005`;
- validation LAV MAE must not degrade;
- validation MissingMacro MAE must not degrade;
- at least two epochs improve validation J by at least `0.003`.

Only after this gate passes is the test loader constructed and traversed once.
The test gate passes when either:

- test J gain is at least `0.005`; or
- test J gain is positive and no individual LAV/LA/LV/L MAE degrades by more
  than `0.002`.

Only a passing seed 1111 authorizes seed 1114. Seed 1114 reuses the selected
lambda and does not repeat the lambda screen.

## Hard stops

- No SAM, router, expert, test-time adaptation, checkpoint soup or ensemble is
  added in this stage.
- No coefficient outside the frozen three-point set is allowed.
- A validation-gate failure ends Video-VREx without test evaluation.
- A frozen-test failure ends Video-VREx without a new parameter search.
