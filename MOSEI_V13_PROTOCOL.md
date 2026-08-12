# MOSEI Frozen v13 Port

## Purpose

This stage ports the already-frozen MOSI v13 complementary expert to MOSEI. It is not a new hyperparameter search and it does not decide whether v13 should exist based on MOSEI Valid. The downstream blend weight remains the MOSI-frozen `0.5 Raw5 + 0.5 v13`.

## Frozen algorithm

The port keeps the v13 mechanism:

1. Development seed is fixed to `1113`.
2. S0 is initialized from the clean DLF seed1113 checkpoint and then frozen.
3. The seed1113 validation-best Stage1 ModDrop checkpoint supplies the counterfactual baseline.
4. Compatibility comes from the formal seed1113 Train-only MOSEI CFCompat cache.
5. Train is divided into five deterministic video-grouped folds.
6. Each fold trains only a v8-compatible residual head bank.
7. The objective is the v4 supervised missing + DISTILL/PRESERVE objective separated into supervised and selective gradient components.
8. The v12 asymmetric gradient surgery is applied once per original optimizer update window.
9. Adam proposes its actual parameter displacement; v13 projects that displacement against the per-mode Teacher-beneficial and S0-beneficial Train-only sentinel MAE halfspaces.
10. Fold checkpoint selection is the frozen earliest epoch within 1% of the absolute-best Train-video-holdout J.
11. Final inference uses 4-of-5 same-sign consensus and the median residual, otherwise zero correction.

The inherited frozen constants such as the 0.02 Teacher-beneficial/S0-beneficial margins, residual hidden dimension, residual bound, five folds, 4-of-5 consensus, Adam optimizer, update window, and 1%-near-optimal selector are not retuned on MOSEI.

## MOSEI-specific engineering changes

The old MOSI v13 entrypoint is not called. The new `train_mosei_v13.py` reuses only the algorithmic fold-training functions and removes MOSI-only v12/v10/v8/v4 historical artifact gates.

Windows uses `num_workers=0`. The MOSEI config remains batch size 16, update_epochs 10, learning rate 1e-4, scheduler patience 5, and early_stop 10. The current environment uses the same `matmul_precision=high` convention as the frozen MOSEI Stage0/Stage1/CFCompat runs.

The historical initial LAV Teacher/Student equivalence assertion is performed on Train instead of Valid, so Official Valid is not evaluated before all five Train-only fold banks are frozen.

The full frozen consensus `state_dict` is saved explicitly. This is required for the later single final MOSEI Test so the v13 S0 missing-token/adapter state does not need to be reconstructed from RNG.

## Split protocol

During fold training and fold checkpoint selection:

- Train is used.
- Train-video holdouts are used.
- Official Valid is not evaluated.
- Official Test is not constructed.

After all five fold banks are frozen, Official Valid is evaluated once for characterization and to produce the v13 Valid prediction file needed by the already-frozen 50/50 blend. No v13 hyperparameter, fold checkpoint, blend weight, or architecture choice may change from this Valid result.

Official Test remains untouched until the complete `Raw5 + v13 fixed 0.5/0.5 + DP57` method is frozen.

## Required upstream assets

For seed1113:

- `pt/DLF_mosei_seed1113_best.pth`
- `pt/missing_baseline/moddrop/DLF_mosei_seed1113_best.pth`
- the formal `mosei_cf_compat_v1` Train-only compatibility cache bound by Stage1 evaluator SHA

The five Raw5 Valid member CSVs should also already exist before this stage so the downstream blend can run immediately after v13 finishes.

## Outputs

Formal outputs are under:

- `result/missing_baseline/cfcompat_adam_step_safety_v13/mosei/valid_screen/seed1113_dev/`
- `pt/missing_baseline/cfcompat_adam_step_safety_v13/mosei/valid_screen/seed1113_dev/`

Important artifacts include:

- `v13_valid_predictions.csv`
- `adam_step_safety_v13_candidate_grid.csv`
- fold assignment, fold metrics, Train decisions, surgery-window diagnostics, actual-step safety diagnostics, and sentinel manifests
- `adam_step_safety_v13_mosei_summary.json`
- `frozen_consensus_valid_ready.pth`

Smoke outputs are isolated under `smoke/` and never overwrite formal artifacts.

## Commands

Preflight:

```powershell
.\scripts\run_windows_mosei_v13.ps1 -Preflight
```

Smoke (five folds, at most two epochs per fold):

```powershell
.\scripts\run_windows_mosei_v13.ps1 -Smoke
```

Formal:

```powershell
.\scripts\run_windows_mosei_v13.ps1
```

After formal v13 completes, the next stage is not a blend search. It is exactly:

`Raw5 0.5 + v13 0.5 -> DP57 with the already Valid-selected Raw5 anchor seed1114`.
