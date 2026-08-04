# CFCompatKD + Mode-Consistent C-Mixup v1

## Purpose

This stage transfers the mature C-Mixup regression augmentation into the official
MOSI CFCompatKD training path. It targets small-sample continuous-regression
generalization. It does not introduce a new inference architecture.

## Frozen base

- Base branch: `experiment/cfcompat-distillation-evidence-v1`.
- Dataset/protocol: official MOSI train/valid/test.
- Initial Student and frozen Teacher: the matched clean Gate-3 checkpoint.
- Evaluator and compatibility cache: the locked Stage-3 seed-matched artifacts.
- Original objective remains:
  `L_full + L_missing + L_CFCompatKD`.
- Missing-mode sequence, data order, optimizer, scheduler, accumulation, and
  validation checkpoint rule remain the Stage-3 definitions.

## C-Mixup adaptation

C-Mixup is applied to the differentiable input of the original DLF `proj1`.
The DLF source and inference path are not modified.

For each train batch:

1. run the original full LAV view and capture its fusion feature;
2. sample the original per-sample LA/LV/L masks and capture the missing-view
   fusion feature;
3. inside each realized missing-mode group only, sample a partner using the
   label-KDE probability
   `p(j|i) proportional to exp(-(y_i-y_j)^2/(2 sigma^2))`;
4. draw `lambda ~ Beta(alpha, alpha)`;
5. use the same partner and lambda for the full and missing fusion features;
6. pass both mixed features through the unchanged DLF final residual MLP;
7. apply regression loss to the interpolated label.

Frozen values:

- `alpha = 2.0`
- `sigma = 0.5` sentiment-label units
- `mix_weight = 1.0`
- self-pairs are excluded when a mode group has at least two samples;
- singleton mode groups are omitted from the mixed loss;
- the extra dropout forward preserves and restores global RNG state.

Mixed samples never receive a Teacher prediction, compatibility value, KD loss,
or synthetic auxiliary-head target.

## Test lock

Training constructs train and valid only. The test loader is constructed after
the validation-best checkpoint is frozen and is traversed once. Test cannot
select the epoch, method, bandwidth, alpha, or loss weight.

## Seed gate

Seed 1111 is run first. Promotion requires all of:

- Valid J gain over locked CFCompatKD at least `0.005`;
- Valid LAV MAE gain at least `0.005`;
- Valid MissingMacro MAE degradation no greater than `0.002`;
- at least two epochs beat baseline Valid J by `0.003`.

Only a passing seed 1111 authorizes seed 1114. Only two passing seeds authorize
five-seed replication. No test result participates in these gates.

## Method boundary

This stage adds no inference parameters, module, Teacher, router, expert,
ensemble, residual correction, second-stage prediction, or extra input.
