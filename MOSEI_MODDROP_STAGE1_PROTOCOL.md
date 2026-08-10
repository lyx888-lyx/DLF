# MOSEI Stage-1 DLF-ModDrop protocol v1

This stage ports the already frozen Stage-1 ModDrop mechanism from MOSI to MOSEI.
It is a prerequisite for the later CFCompat evaluator/cache.  It does not use
or construct the official MOSEI Test split during training.

## Inputs

Each seed `1111..1115` must have its SHA-bound clean DLF validation-best source:

```text
pt/DLF_mosei_seedSEED_best.pth
result/missing_baseline/mosei_clean_stage0_v1/seedSEED/run_manifest.json
```

The Stage-1 runner refuses to train when the checkpoint SHA differs from the
Stage-0 manifest.

## Frozen ModDrop mathematics

Modality order remains `[text, audio, vision]`; text is always present.
Each training sample independently receives one missing view from `LA`, `LV`, or
`L`, each with probability `1/3`, using the dedicated RNG seed `seed + 104729`.

The clean DLF checkpoint is loaded first. Stage-1 adds only:

- a learnable missing-audio token;
- a learnable missing-vision token;
- a zero-initialized, bias-free modality-mask residual adapter.

The objective is unchanged:

```text
L_total = L_full + 1.0 * L_missing
```

`L_full` is the original complete-view DLF combined loss. `L_missing` is the
original five-head task loss only.

## Validation-only selection

Every epoch evaluates `LAV`, `LA`, `LV`, and `L` on official Valid. The sole
checkpoint criterion is

```text
J_val = 0.5 * MAE_LAV + 0.5 * mean(MAE_LA, MAE_LV, MAE_L)
```

The scheduler also receives `J_val`. Formal training keeps the configured MOSEI
`batch_size=16`, `update_epochs=10`, `learning_rate=1e-4`, `patience=5`, and
`early_stop=10`. There is no formal `max_epochs` cap, because adding one would
change the validation-best selection protocol. `max_epochs` is available only
for isolated smoke runs.

Official Test is not constructed, read, reported, or used for tuning here.

## 8 GiB Windows execution adaptation

The reference trainer constructs the full-view and missing-view graphs before a
single backward on their sum. The MOSEI Windows runner instead performs

```text
backward(L_full)
backward(L_missing)
# no optimizer step occurs between them
```

before the same gradient-accumulation boundary. Therefore the accumulated
parameter gradient remains

```text
grad(L_full) + grad(L_missing) = grad(L_full + L_missing)
```

while the two activation graphs do not need to be resident simultaneously. This
is an execution-memory adaptation only; the objective, optimizer-step boundary,
checkpoint selection, scheduler metric, missing-view RNG, and early stopping are
unchanged.

Formal Windows runs fix `num_workers=0` and default to
`torch.set_float32_matmul_precision("high")`, matching the clean MOSEI Stage-0
local execution policy.

## Canonical downstream artifacts

Formal validation-best checkpoints:

```text
pt/missing_baseline/moddrop/DLF_mosei_seedSEED_best.pth
```

The canonical Stage-1 CSV consumed by later CFCompat is incrementally upserted:

```text
result/missing_baseline/moddrop/train/mosei_per_seed.csv
result/missing_baseline/moddrop/train/mosei_summary.csv
```

Per-seed SHA/environment manifests are written under:

```text
result/missing_baseline/mosei_moddrop_stage1_v1/seedSEED/run_manifest.json
```

## Commands

Preflight:

```powershell
.\scripts\check_windows_mosei_moddrop_stage1_v1.ps1
```

Isolated two-epoch smoke run for seed 1111:

```powershell
.\scripts\run_windows_mosei_moddrop_stage1_v1.ps1 -Seed 1111 -Smoke
```

Formal seed 1111:

```powershell
.\scripts\run_windows_mosei_moddrop_stage1_v1.ps1 -Seed 1111
```

Run one seed per process on the laptop GPU. Do not run official Test as part of
Stage-1 development.
