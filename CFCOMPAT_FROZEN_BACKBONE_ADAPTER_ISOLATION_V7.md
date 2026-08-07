# CFCompatKD v7 — Frozen-Backbone Missing-Adapter Isolation

## Question

v5.2 showed that many Teacher-beneficial Valid clips start from a strong initial
Student (S0) but end in negative transfer, often across LA/LV/L together. v6
showed that a sample-local, Train-label-conditioned S0 trust loss does not solve
that drift. v7 therefore asks a harder causal question:

> If shared DLF parameters are not allowed to move at all, can missing-specific
> parameters retain S0's useful function while still repairing difficult cases?

## Frozen protocol

- Dataset: MOSI.
- Development seed: 1113 only.
- Decision split: official Valid only.
- Official Test: never constructed or accessed.
- New Student trajectories: exactly one.
- Checkpoint selection: minimum official Valid J.
- Base loss/routing: v4 DISTILL / PRESERVE / ABSTAIN unchanged.
- No v6 S0 trust term.
- No v5 gate classifier or threshold tuning.

## Hard isolation

The existing `MissingModalityWrapper` is partitioned as:

- frozen shared function: `student.backbone`;
- trainable missing-specific parameters:
  - `missing_audio_token`,
  - `missing_vision_token`,
  - `mask_adapter.weight`.

All other parameters have `requires_grad=False`. During training the shared
backbone is forced to `eval()` so buffers/dropout state cannot create hidden
functional drift. The optimizer is constructed only from the three expected
missing-specific parameter tensors.

A SHA-256 digest of the entire backbone state dict (parameters + buffers) is
recorded before training, checked after every epoch, and checked again after
loading the validation-best checkpoint. Any change aborts the run.

## Why this is stronger than v6

v6 allowed all shared parameters to update and tried to correct selected Train
points after they regressed from S0. That was still a local loss. v7 removes the
shared update path itself. If v7 protects Teacher/S0-beneficial cases, the
shared-update interference hypothesis gains support. If v7 merely collapses to
S0 and cannot repair harmful cases, the next question is adapter capacity.

## Per-epoch Valid trajectory

Epoch 0 is the frozen initial Student. For every subsequent epoch v7 saves
sample-level predictions for LAV/LA/LV/L. The trajectory artifact derives:

- gain vs frozen ModDrop baseline;
- Teacher advantage;
- S0 gain;
- absolute drift from S0;
- positive / negative / severe-negative transfer;
- whether the prediction crossed the baseline from the S0 side;
- whether final error is worse than S0 by 0.02.

Additional artifacts report the first epoch at which every sample×mode becomes
negative/severe/crossed, plus per-epoch counts of clips for which all three
missing modes fail together.

## Pre-registered mechanism checks

No grid is trained. The single v7 candidate is compared with frozen v4 using:

- Valid J degradation <= 0.002;
- Teacher-beneficial NTR reduction >= 0.05;
- Teacher-nonbeneficial NTR degradation <= 0.03;
- overall NTR degradation <= 0.01.

These thresholds are a mechanism signal, not a claim of final model selection.

## Main outputs

Under
`result/missing_baseline/cfcompat_frozen_backbone_adapter_isolation_v7/mosi/valid_screen/seed1113_dev/`:

- `frozen_backbone_v7_valid_screen_summary.json`
- `frozen_backbone_v7_transfer_summary.csv`
- `frozen_backbone_v7_valid_epoch_predictions.csv`
- `frozen_backbone_v7_valid_epoch_event_trajectory.csv`
- `frozen_backbone_v7_valid_epoch_transfer_summary.csv`
- `frozen_backbone_v7_valid_failure_onset.csv`
- `frozen_backbone_v7_valid_clip_failure_epoch_summary.csv`
- `frozen_backbone_v7_trainable_parameter_delta.csv`
- `frozen_backbone_v7_train_decisions.csv`

## Interpretation

A positive result supports hard shared-parameter isolation as the mechanism.
A safe-but-underpowered result motivates a larger residual/low-rank adapter
while keeping the backbone frozen. A negative result despite exact backbone
immutability weakens the hypothesis that shared-backbone drift is the main
remaining cause and redirects attention to missing-specific optimization or
prediction-level conflicts.
