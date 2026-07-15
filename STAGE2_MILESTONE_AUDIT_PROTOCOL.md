# Stage 2.5 Counterfactual and Milestone Audit Protocol

## Scope

This branch audits only frozen MOSI seed-1111 models.  It creates no optimizer,
performs no backward pass, changes no checkpoint, and does not train, tune, select
a new epoch, or implement Stage 3.  Every model is set to evaluation mode and all
forwards run under \`torch.inference_mode()\`.

The audit compares three distinct checkpoints:

- Gate 3 plain DLF, evaluated by DirectMask input zeroing;
- Stage 1 \`MissingModalityWrapper(DLF)\` ModDrop;
- Stage 2 \`MissingModalityWrapper(DLF)\` FixedKD, with no teacher at inference.

For every model the audit records its path, SHA-256, byte size, and state-dict
key count.  Parameters are frozen for the audit and are checked to retain
\`grad is None\`.

## Four fixed input modes

The modality order is permanently text/audio/vision.  LAV is complete input; LA
zeros or masks vision; LV zeros or masks audio; and L zeros or masks both audio
and vision.  The single-split loader is always non-shuffled.  Original dataset
sample IDs are retained, with zero-based stable indices retained alongside them.

## Validation gate

Validation runs before any test access.  It recomputes the four-mode metrics and
compares them to the frozen DirectMask, ModDrop, and FixedKD CSV records using an
absolute tolerance of 1e-6 for every Acc/F1/Corr/MAE/Loss field.  A mismatch
raises immediately and prevents a test audit.

## Fixed descriptive statistics

For ModDrop and FixedKD, mode differences are
\`abs(pred_LAV - pred_mode)\`.  The report contains mean, standard deviation,
median, p75/p90/p95, maximum, fixed threshold fractions, and sign flips using
the unchanging rule \`prediction > 0\` versus \`prediction <= 0\`.

FixedKD-to-ModDrop mean-difference ratios are described as compressed below
0.75, similar from 0.75 through 1.25, and amplified above 1.25.  These labels
are descriptive only and never affect a model or parameter.

Error changes use
\`abs(pred_mode-label) - abs(pred_LAV-label)\`.  Improved is below \`-1e-8\`,
worsened is above \`1e-8\`, and unchanged is within that closed tolerance.

## Counterfactual decomposition

The primary frozen evaluator is ModDrop.  For each sample:

\`\`\`
r_A  = F_LA - F_L
r_V  = F_LV - F_L
r_AV = F_LAV - F_LA - F_LV + F_L
\`\`\`

The script rejects an audit unless each sample satisfies the reconstruction and
three missing-contribution identities to a maximum error of 1e-6.  Contributions
and their error associations are descriptive and cannot change training.

## One-time test milestone

Test mode requires \`--confirm-stage2-milestone-test\`.  It refuses execution if
the final test directory or its \`AUDIT_COMPLETED.json\` already exists.  Output
is first produced in a temporary sibling directory, moved atomically only after
all checks pass, and finally locked by an atomic completion marker.  A failed
test leaves its failure text in the temporary audit directory and creates no
completion marker.

Generated audit results are ignored by Git.  The implementation and tests must
be committed and pushed after validation passes and before the one permitted
test audit.
