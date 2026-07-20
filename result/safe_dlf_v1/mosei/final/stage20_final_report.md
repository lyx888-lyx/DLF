# Stage 20 SAFE-DLF v1 final report

- Status: `STAGE20_SAFE_DLF_SEED1_FAILED`, `STAGE20_ROUTE_CLOSED`
- Retained: **none**
- Locked Test access count: **0**
- Selection split: **official Valid only**
- Seed 1114: **not run** (seed1111 gate failed)

## Phase R: Stage19 checkpoint retrospective

- Uniform best: epoch 2, J `0.515493`.
- MGD best among all existing Valid checkpoints: epoch 8, J `0.532166`.
- MGD - Uniform delta J: `+0.016672` (positive is worse); status: `STAGE19_RETRO_MGD_CONFIRMED_CLOSED`.
- Therefore the Stage19 MGD best checkpoint is **not** genuinely better than the Uniform best checkpoint; MGD was not retrained and MGRD was not run.

## Phase A: frozen Uniform ghost audit

- Maximum GAR: `1.000962`
- Raw-input filler sensitivity with fixed availability: `0.000e+00` (no measurable filler sensitivity).
- Aggregate unsupported gradient share: `0.175387` (LA `0.108396`, LV `0.136396`, L `0.281369`).
- Absent-branch parameter gradient ratio: LA `0.041685`, LV `0.061603`, L `0.075850`.
- Frozen LFA attention mass assigned to an absent key/value branch reached `1.0`.
- Interpretation: ghost activation/backward support mismatch is real, even though replacing the raw filler does not change the wrapped model output.

## Seed 1111 selected-checkpoint comparison

| Method | Epoch | J | Delta J | Missing modes MAE improved | Passed | Wall h |
|---|---:|---:|---:|---:|---|---:|
| SAO-PM | 9 | 0.521334 | +0.005840 | 0/3 | False | 1.390 |
| SAFE Full | 9 | 0.518769 | +0.003276 | 1/3 | False | 1.410 |

SAFE Full - SAO-PM delta J: `-0.002565`.

Lower J/MAE is better; higher Corr/Acc/F1 is better. All values below are recomputed from the frozen selected Valid checkpoints.

| Mode | Metric | Uniform | SAO-PM | PM - Uniform | SAFE Full | Full - Uniform |
|---|---|---:|---:|---:|---:|---:|
| LAV | MAE | 0.514148 | 0.520997 | +0.006850 | 0.520006 | +0.005858 |
| LAV | Corr | 0.750748 | 0.742248 | -0.008500 | 0.745307 | -0.005441 |
| LAV | Acc7 | 0.545697 | 0.539818 | -0.005879 | 0.534474 | -0.011224 |
| LAV | Acc5 | 0.560663 | 0.553180 | -0.007483 | 0.548370 | -0.012293 |
| LAV | Acc2 | 0.849791 | 0.842837 | -0.006954 | 0.845619 | -0.004172 |
| LAV | F1 | 0.846929 | 0.844266 | -0.002663 | 0.845892 | -0.001037 |
| LA | MAE | 0.519748 | 0.521791 | +0.002044 | 0.517328 | -0.002419 |
| LA | Corr | 0.746631 | 0.741496 | -0.005135 | 0.744468 | -0.002164 |
| LA | Acc7 | 0.544094 | 0.540887 | -0.003207 | 0.533939 | -0.010155 |
| LA | Acc5 | 0.559059 | 0.553715 | -0.005345 | 0.547835 | -0.011224 |
| LA | Acc2 | 0.849791 | 0.841446 | -0.008345 | 0.848401 | -0.001391 |
| LA | F1 | 0.846024 | 0.842778 | -0.003246 | 0.848401 | +0.002376 |
| LV | MAE | 0.512607 | 0.521046 | +0.008439 | 0.516979 | +0.004371 |
| LV | Corr | 0.750533 | 0.742062 | -0.008471 | 0.745001 | -0.005532 |
| LV | Acc7 | 0.551577 | 0.543025 | -0.008552 | 0.537146 | -0.014431 |
| LV | Acc5 | 0.566007 | 0.555852 | -0.010155 | 0.551577 | -0.014431 |
| LV | Acc2 | 0.850487 | 0.842837 | -0.007650 | 0.847705 | -0.002782 |
| LV | F1 | 0.848510 | 0.844157 | -0.004353 | 0.847531 | -0.000980 |
| L | MAE | 0.518163 | 0.522173 | +0.004011 | 0.518290 | +0.000127 |
| L | Corr | 0.746564 | 0.741166 | -0.005398 | 0.743048 | -0.003517 |
| L | Acc7 | 0.545163 | 0.538749 | -0.006414 | 0.533939 | -0.011224 |
| L | Acc5 | 0.559594 | 0.551577 | -0.008017 | 0.548370 | -0.011224 |
| L | Acc2 | 0.851182 | 0.842142 | -0.009040 | 0.849791 | -0.001391 |
| L | F1 | 0.847857 | 0.843384 | -0.004473 | 0.849584 | +0.001727 |
| MissingMacro | MAE | 0.516839 | 0.521670 | +0.004831 | 0.517532 | +0.000693 |
| MissingMacro | Corr | 0.747910 | 0.741575 | -0.006335 | 0.744172 | -0.003737 |
| MissingMacro | Acc7 | 0.546945 | 0.540887 | -0.006057 | 0.535008 | -0.011937 |
| MissingMacro | Acc5 | 0.561554 | 0.553715 | -0.007839 | 0.549261 | -0.012293 |
| MissingMacro | Acc2 | 0.850487 | 0.842142 | -0.008345 | 0.848632 | -0.001854 |
| MissingMacro | F1 | 0.847464 | 0.843440 | -0.004024 | 0.848505 | +0.001041 |

## Seed1111 gates

### SAO-PM

| Gate | Result |
|---|---|
| at_least_two_missing_modes_mae_improved | FAIL |
| classification_safe | FAIL |
| delta_J_le_minus_0_003 | FAIL |
| lav_corr_safe | FAIL |
| lav_mae_safe | FAIL |
| missing_macro_corr_safe | FAIL |
| missing_macro_mae_improved | FAIL |
| no_extra_inference_forward | PASS |
| not_single_mode_driven | FAIL |

### SAFE Full

| Gate | Result |
|---|---|
| absent_gradient_le_1e_8 | PASS |
| at_least_two_missing_modes_mae_improved | FAIL |
| classification_safe | FAIL |
| delta_J_le_minus_0_003 | FAIL |
| filler_sensitivity_le_1e_6 | PASS |
| lav_corr_safe | FAIL |
| lav_mae_safe | FAIL |
| lav_parity | PASS |
| missing_macro_corr_safe | FAIL |
| missing_macro_mae_improved | FAIL |
| no_extra_inference_forward | PASS |
| not_single_mode_driven | FAIL |
| unsupported_gradient_share_zero | PASS |

Both candidates fail the frozen seed1111 promotion gate. SAFE Full is better than SAO-PM on J (`-0.002565`), but remains worse than Uniform and violates the regression/correlation/classification safety gates.

## SAFE Full implementation evidence

| Check | Observed | Required | Result |
|---|---:|---:|---|
| LAV output max abs diff | 0.000e+00 | <= 1e-6 | PASS |
| LAV total-loss abs diff | 0.000e+00 | <= 1e-6 | PASS |
| LAV component-loss max diff | 0.000e+00 | <= 1e-6 | PASS |
| Filler output max abs diff | 0.000e+00 | <= 1e-6 | PASS |
| Filler fusion max abs diff | 0.000e+00 | <= 1e-6 | PASS |
| Absent input gradient max | 0.000e+00 | <= 1e-8 | PASS |
| Absent representation max | 0.000e+00 | <= 1e-8 | PASS |
| Unsupported loss max | 0.000e+00 | 0 | PASS |
| Present input gradient min | 0.870774 | > 0 | PASS |

The selected checkpoint passed **18/18** implementation tests; failed tests: **0**.

## Required closure

1. **Stage19 MGD vs Uniform:** MGD is worse (`Delta J = +0.016672`); not retained.
2. **Baseline ghost activation:** yes; maximum GAR `1.000962`.
3. **Baseline filler sensitivity:** no measurable raw-filler sensitivity; max prediction difference `0.000e+00`.
4. **Unsupported gradient share:** aggregate `0.175387` (17.54%).
5. **PM improvement:** no; `Delta J = +0.005840`, 0/3 missing-mode MAEs improved.
6. **Full SAFE improvement:** no; `Delta J = +0.003276`, 1/3 missing-mode MAEs improved.
7. **Full vs PM:** Full has lower J by `0.002565`, but neither passes and Full has classification trade-offs.
8. **LAV parity:** passed; output/loss/component maximum differences are all zero.
9. **Filler invariance:** passed; prediction and fusion differences are zero.
10. **Absent gradient:** passed; absent input gradient and unsupported loss are zero.
11. **Seed1111:** failed the frozen promotion gate.
12. **Seed1114:** not run, by preregistered gate.
13. **Final retention:** none.
14. **Single-seed wall time:** SAO-PM `1.390` h; SAFE Full `1.410` h.
15. **Branch/commit/push/clean:** branch `experiment/mosei-safe-dlf-v1`; implementation commit `c1f686e8164f729e0e31d6ad18cec21d111b0c3d`. Final push and clean-state verification are recorded in the operator handoff.
16. **Tests:** 18/18 passed on the selected SAFE Full checkpoint.
17. **GPU release:** both training jobs exited; final GPU process verification is recorded in the operator handoff.
18. **Locked Test access:** `0`.
19. **Dependencies:** none upgraded.
20. **Original worktrees:** not modified.

## Decision

`STAGE20_SAFE_DLF_SEED1_FAILED` and `STAGE20_ROUTE_CLOSED`.

The implementation/mechanism audit succeeded, but the selected Valid metrics did not. No coefficient, learning-rate, batch-size, AMP, KD, Test-based rescue, seed1114 run, or cross-dataset run was attempted.
