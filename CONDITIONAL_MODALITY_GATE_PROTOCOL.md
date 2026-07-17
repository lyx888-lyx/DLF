# Stage 7B Conditional Modality Utility Gating Protocol

## Frozen provenance

- Base branch: `analysis/modality-utility-audit-v1`
- Base commit: `1e5ab2430307576127ad0845fde32b433a0da744`
- Seed: 1111 only
- Fixed reference: CFCompatKD validation-best epoch 9,
  J_valid 0.677963724, J_test 0.717881143.
- Teacher, Student backbone, optimizer, scheduler, missing-mask RNG,
  compatibility cache, and CFCompatKD formula remain the Stage3 definitions.

## Evidence gate

Stage7A train/valid raw CSVs define stable audio and vision groups. A positive
sample has positive Gate3 gain, Gate3 shuffle damage, CFCompat gain, and
CFCompat shuffle damage. A negative sample has all four values less than or
equal to zero. Every other sample is ambiguous and is excluded from gate
supervision.

Training requires at least 32 train positives, 32 train negatives, 8 valid
positives, and 8 valid negatives per modality. Valid groups are diagnostics
only. Test cannot occur in the utility artifact or target table.

## Gate

The real DLF `proj_l`, `proj_a`, and `proj_v` outputs are the projected
pre-fusion representations. A qualified modality receives one linear gate over
the concatenated mean-pooled text and modality projections. Its probability is
`q=sigmoid(z)` and its scale is `g=2q`. Weight and bias are zero initialized, so
q=0.5 and g=1 exactly. New gate construction restores the RNG state so Stage3's
dropout sequence is unchanged.

The projected audio or vision tensor is multiplied by its sample scalar before
the modality-specific and shared encoders. Text is never gated. The wrapper's
modality-present mask forces scale one for a missing modality, so missing tokens
are never utility gated. LAV uses both eligible gates, LA only audio, LV only
vision, and L neither.

## Losses

Train reliable-positive targets are one and reliable-negative targets are zero.
Ambiguous and valid samples never enter BCE. Utility BCE uses q on the correct
full-modality path and is averaged over qualified modalities.

For `utility_gate_matched`, train reliable-positive samples additionally use the
Stage7A derangement `(epoch-1) mod 10`. Only the relevant modality tensor is
replaced. The constraint is the mean of
`relu(SmoothL1(match,label)-SmoothL1(shuffle,label))`; there is no margin.
The shuffled forward reuses the matched forward's dropout RNG state, and the
training RNG is restored afterward, so the comparison changes only the selected
modality and does not shift Stage3's subsequent random trajectory.

All active loss coefficients are exactly one:

`L_full + L_missing + L_CFKD + L_utility + L_match`.

Identity replay has no trainable gate, utility loss, or matched loss.
Utility-gate has no matched loss.

## Selection and stop rules

Every epoch evaluates valid and test under the original benchmark protocol.
Only validation J selects the main checkpoint. Test-best is diagnostic only,
and inference is Student-only.

Identity Replay must reproduce epoch 9 and the fixed J metrics within 1e-4
before the other formal variants may run. After the three registered variants,
the final report classifies A-F and stops. It does not start another seed or
five-seed replication.

No new Teacher, prediction residual, gradient surgery, adapter/LoRA,
mode-specific head, contrastive loss, reliability/recoverability signal,
temperature, margin, threshold search, or loss-weight search is permitted.
