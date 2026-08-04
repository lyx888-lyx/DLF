# CFCompatKD + SAM v1: locked two-seed Valid screen

## Goal

Test whether Sharpness-Aware Minimization improves the optimization stability of
CFCompatKD without changing samples, modality masks, the Teacher, the
compatibility gate, the network, or inference. This stage is a development
screen only. Official Test is forbidden.

## Frozen base

- Base branch: `experiment/cfcompat-distillation-evidence-v1`.
- Dataset: official CMU-MOSI train and valid splits only.
- Formal seeds: `1111` and `1114`, both required in one run.
- Original Student initialization, frozen Teacher, validation-best Stage-1
  evaluator, and train-only compatibility cache are reused unchanged.
- Original objective remains exactly
  `L_full + L_missing + L_CFCompatKD`.
- Checkpoints are selected only by
  `J = 0.5 * MAE_LAV + 0.5 * mean(MAE_LA, MAE_LV, MAE_L)` on official Valid.

## SAM adaptation

The only candidates are `rho in {0.01, 0.05}` with Adam as the base optimizer.
No other SAM variant or hyperparameter is permitted.

The original DLF configuration uses `update_epochs=10`. Therefore one optimizer
update is defined over the same ten-microbatch accumulation window used by the
baseline:

1. sample each microbatch's missing-modality mask once;
2. accumulate first-pass CFCompatKD gradients over the complete window;
3. perturb parameters once using the global window gradient norm;
4. replay the same microbatches, missing masks, and dropout RNG states;
5. restore parameters and apply one Adam update from second-pass gradients.

The second forward pass is not allowed to consume an additional dropout RNG
trajectory. This keeps the comparison focused on SAM rather than extra random
augmentation.

## Selection and gate

Both rho values run on both formal seeds. Rho is selected by the lowest mean
Valid J across the two seeds; ties prefer the smaller rho.

The selected rho passes only when all conditions hold:

- both seeds have positive Valid J gain;
- mean two-seed Valid J gain is at least `0.005`;
- no seed and no LAV/LA/LV/L Valid MAE degrades by more than `0.002`;
- LAV and MissingMacro do not materially degrade on either seed;
- each seed has at least two epochs improving Valid J by at least `0.003`.

## Test lock and next stage

This branch never constructs official Test, even when the dual-seed gate passes.
A pass authorizes only a separate train-only grouped stability screen. Official
Test remains locked until that additional screen is implemented and passes.

## Hard stops

- No rho outside `{0.01, 0.05}`.
- No learning-rate, batch-size, dropout, scheduler, loss, or checkpoint-rule
  adjustment.
- No ASAM, GSAM, LookSAM, SAM warm-up, router, expert, Mixup, V-REx, or video
  sampler.
- Dual-seed Valid failure stops SAM.
- Test results may not be used to develop this method.
