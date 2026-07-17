# Stage 8: Stability-First CFCompatKD

## Locked scope

- Base branch/commit: `experiment/cf-compat-kd-multiseed-v1` at
  `276fc26e6e7d1998f7dd746edfca1efbbb691e55`.
- Seeds and execution order: 1111, 1112, 1113, 1114, 1115.
- The online Student, Full Teacher, train-only compatibility cache, loss,
  optimizer, scheduler, batch size, early stopping, missing generator, and
  validation objective are the unchanged Stage 3 CFCompatKD protocol.
- Main checkpoints and the global strategy use validation J only.
- No later Stage 4–7 code, new loss, new network, new Teacher, cross-seed
  weight averaging, greedy soup, fine-tuning, or test-based selection is used.

## EMA

- Decay is fixed at 0.999.
- EMA starts as an exact deep copy of the initial Student.
- Exactly one EMA update follows every online `optimizer.step()`.
- Named parameters use `0.999 * ema + 0.001 * online`; every registered buffer
  is copied exactly from the online Student after each optimizer step.
- EMA is frozen, absent from optimizer/backward, and evaluated in `eval()` mode.
- All EMA initialization/evaluation work restores Python, NumPy, Torch CPU, and
  CUDA RNG states.
- The EMA checkpoint is selected only by EMA validation J.

The real DLF/MissingModalityWrapper has no BatchNorm. Static transformer
`_float_tensor` device/dtype anchors are audited and copied, never averaged or
included in parameter-distance evidence. No BN recalibration or additional
running-stat policy is introduced.

## Trajectory Soup

- The five online Student checkpoints with lowest validation J are retained.
- Ranking is `(J_valid ascending, epoch ascending)`.
- Top-3 and Top-5 same-seed uniform soups are built after training.
- Only online Student states are sources. EMA/Teacher/test/cross-seed states are
  forbidden.
- Floating tensors are averaged on CPU in FP64 then cast to the source dtype.
- Keys, shapes, and dtypes must match. Non-floating tensors must be identical.
- Soup is evaluated once on valid/test and is never fine-tuned.

## Replay and stopping gates

Every online trajectory is compared with its locked Stage 3 validation-selected
result. Best epoch must match; validation J, test J, and four test-mode MAEs must
be within `1e-4`. The exact dedicated-generator missing-mode sequence SHA is
also checked. Seed1111 is run first. Any replay, RNG isolation, state integrity,
NaN/Inf/OOM, or validation-selection failure stops the stage.

## Global selection and reporting

After all five replays pass, one strategy is selected globally:

`argmin mean_seed(J_valid)` over EMA, Soup-3, Soup-5, with tie priority
EMA > Soup-3 > Soup-5. Online is only the baseline. Test results do not select
the strategy.

The five online validation-best prediction mean is reported separately as an
equal-weight multi-model inference upper bound. It never participates in the
main strategy selection.

Paired results report mean, sample standard deviation, median, min/max,
improved-seed count, deterministic paired-bootstrap 95% CI, and a descriptive
paired t-test. No significance claim is made at n=5.
