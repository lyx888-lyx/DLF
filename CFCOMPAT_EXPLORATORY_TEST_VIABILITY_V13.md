# CFCompatKD Exploratory MOSI Test Viability Probe through v13

## Status

This is **not** a pristine final Test evaluation.

MOSI Test was already accessed historically for an Original-CFCompatKD-vs-v4
viability check.  This probe is an explicitly exploratory second access used
only to decide whether the current mechanism line shows enough cross-split
signal to justify additional research effort.

The required labels are:

- `EXPLORATORY_TEST_VIABILITY_ONLY`
- `TEST_ALREADY_HISTORICALLY_ACCESSED`
- `NO_TEST_DRIVEN_TUNING_ALLOWED`

Any later method change must be justified from Train/OOF/Valid evidence, not
from sample-level Test behavior.  A future strict final claim requires a fresh
untouched external dataset/split.

## Frozen methods

One invocation evaluates exactly these already-frozen methods in fixed order:

1. **Original CFCompatKD v1** — the historical first innovation, bound from the
   Seed1113 `cfcompat_replay` source record in the v2 source manifest.  Only its
   validation-best checkpoint is accepted; any `best_test` or `diagnostic`
   checkpoint is rejected.
2. **Regret-Preserve v4** — validation-best checkpoint recorded by the v4
   candidate grid.
3. **Sample-conditioned residual v8** — validation-best checkpoint recorded by
   the v8 candidate grid.
4. **Gradient-surgery v12** — five Train-video-holdout conservative fold banks,
   combined by the frozen 4-of-5 sign/median consensus rule.
5. **Adam-step functional-safety v13** — five Train-video-holdout conservative
   fold banks, combined by the same frozen consensus rule.

The validation-best ModDrop evaluator from the Original CFCompat source record
is the common missing-modality baseline.  The frozen full-modality Teacher from
that same source record defines Teacher-beneficial Test events.

## No training or Test selection

The probe performs no:

- training;
- checkpoint selection;
- checkpoint averaging;
- gate fitting;
- calibration;
- threshold search;
- inference-rule change;
- Test-selected model loading.

Checkpoint SHA-256 values are verified before evaluation and written to an
aggregate checkpoint manifest.

## Test outputs

No sample-level Test CSV, prediction file, event file, clip ID list, or failure
list is written.

Only aggregate artifacts are persisted:

- `exploratory_test_viability_v13_comparison.csv`
- `exploratory_test_viability_v13_transfer.csv`
- `exploratory_test_viability_v13_checkpoint_manifest.json`
- `exploratory_test_viability_v13_summary.json`

For each method the comparison table reports:

- Test `J = 0.5 * LAV_MAE + 0.5 * mean(LA_MAE, LV_MAE, L_MAE)`;
- LAV/LA/LV/L MAE and missing-mode macro MAE;
- overall mean gain versus frozen ModDrop;
- overall NTR, severe NTR and positive-transfer rate;
- Teacher-beneficial NTR, severe NTR and positive-transfer rate;
- Teacher-nonbeneficial NTR and severe NTR;
- deltas/reductions versus Original CFCompatKD v1.

NTR uses the historical `0.02` margin and severe NTR uses the historical `0.10`
margin.  Teacher-beneficial uses the historical `0.02` Teacher advantage margin.

## Pre-frozen route decision

The primary reference is Original CFCompatKD v1.

The route label is descriptive and is frozen before this Test access:

- `CLEAR_POSITIVE_GENERALIZATION_SIGNAL` iff v13 strictly improves Original
  CFCompat Test J, strictly improves Teacher-beneficial NTR, and does not worsen
  overall NTR.
- `CLEAR_NEGATIVE_GENERALIZATION_SIGNAL` iff none of those three checks pass.
- otherwise `MIXED_GENERALIZATION_SIGNAL`.

No magnitude threshold is tuned from Test.  v13-v12 deltas are also reported
descriptively but do not alter the frozen route rule.

## Interpretation boundary

A positive result means the mechanism line has exploratory cross-split viability
and may justify further non-Test development.  It does **not** restore Test as
an unbiased final endpoint.

A mixed result means some intended mechanism transfers but the overall metric
tradeoff is not resolved.

A negative result is evidence to reduce the priority of this mechanism line
rather than spending many more iterations optimizing only Train/OOF/Valid.
