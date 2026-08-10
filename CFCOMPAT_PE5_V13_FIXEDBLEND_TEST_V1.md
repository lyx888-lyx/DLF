# CFCompat Raw5-PE5 + v13 Fixed-Blend Test v1

## Status

This is an **exploratory/contaminated Test transfer check**, not a pristine final MOSI Test evaluation.

Frozen labels:

- `EXPLORATORY_TEST_FIXED_BLEND_EVALUATION`
- `TEST_ALREADY_HISTORICALLY_ACCESSED`
- `NO_TEST_DRIVEN_WEIGHT_TUNING`
- `FROZEN_WEIGHT_RAW5_0.5_V13_0.5`

## Frozen decision before this Test check

Valid-only evidence:

- Raw5-PE5 `J = 0.666755626598994`
- v13 `J = 0.6675898631413777`
- fixed 50/50 blend `J = 0.6595411400000255`

The fixed grid on Valid was symmetric and broad (`0.25/0.75`, `0.5/0.5`, `0.75/0.25` all improved both parents), and the 0.5/0.5 weight was frozen before this Test check.

No additional blend weights may be evaluated on Test in this experiment.

## Test inputs

### Raw5-PE5

Raw5 is the equal mean of the five previously recovered frozen CFCompat validation-best Test prediction CSVs for seeds 1111--1115. No new Raw5 model Test forward is performed.

The recovered Raw5 Test set must replay `Test J = 0.6976729532082875` within `2e-6` before the blend is allowed.

### v13

v13 is reconstructed from the frozen v8-carried S0 state plus the five frozen v13 conservative residual-bank checkpoints, using the same consensus implementation as the previous exploratory v13 Test probe.

Exactly one new v13 Test model traversal is permitted. Its aggregate metrics must replay the previous aggregate-only exploratory v13 Test comparison within `2e-6` before the blend is allowed.

## Fixed blend

For every Test sample and each mode `LAV`, `LA`, `LV`, `L`:

```text
prediction = 0.5 * Raw5_prediction + 0.5 * v13_prediction
```

The script writes only aggregate metrics. Raw5, v13, and blended sample-level Test predictions remain in memory and are not persisted.

## Verdict

The pre-frozen sign-only verdict is:

- `FIXED_BLEND_GENERALIZATION_SIGNAL_POSITIVE` iff fixed-blend Test `J < Raw5 Test J`.
- otherwise `FIXED_BLEND_GENERALIZATION_SIGNAL_NEGATIVE`.

The Test result must not be used to change the blend weight or introduce a Test-tuned selector.

## Windows command

```powershell
.\scripts\run_windows_cfcompat_pe5_v13_fixedblend_test_v1.ps1 -AcknowledgeExploratoryContaminatedTest
```

Use `-Overwrite` only for a conscious technical rerun after inspecting an existing/failed output directory.
