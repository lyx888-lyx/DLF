# MOSEI CFCompat-v1 protocol

This stage ports the frozen original `DLF-CFCompatKD-v1` member training from
MOSI to MOSEI. These five validation-selected student members are the upstream
members of the later equal-weight Raw5 prediction ensemble.

This port is **Valid-only**. Official MOSEI Test is not constructed, evaluated,
or used for checkpoint selection, tuning, diagnostics, or early stopping.

## Frozen inputs

For each formal seed `1111..1115`:

1. Clean DLF teacher/student initialization is the same seed's Stage-0
   validation-best checkpoint:

   ```text
   pt/DLF_mosei_seedSEED_best.pth
   result/missing_baseline/mosei_clean_stage0_v1/seedSEED/run_manifest.json
   ```

2. The counterfactual compatibility evaluator is the same seed's Stage-1
   ModDrop validation-best checkpoint recorded in:

   ```text
   result/missing_baseline/moddrop/train/mosei_per_seed.csv
   result/missing_baseline/mosei_moddrop_stage1_v1/seedSEED/run_manifest.json
   ```

The runner rejects any Stage-0 or Stage-1 SHA mismatch, dataset/seed mismatch,
Stage-1 best-epoch mismatch, or Stage-1 manifest that says Test was constructed.

## Train-only compatibility cache

Each seed gets its own Train-only counterfactual cache. The expected MOSEI Train
size is exactly `16,326` unique `sample_index` values `0..16325`.

The frozen Stage-1 evaluator is run on `LAV`, `LA`, `LV`, and `L`. For each
missing mode `m`:

```text
delta_m = abs(pred_LAV - pred_m)
rank_m  = stable average rank(delta_m) independently within mode m
q_m     = (rank_m - 0.5) / N
compat_m = 1 - q_m
```

No label enters the compatibility transform. The cache source is Train only.
The formal cache path is:

```text
result/counterfactual_compatibility/mosei_cf_compat_v1/mosei/seedSEED/
    train_counterfactual_compatibility.csv
```

## Frozen CFCompat-v1 objective

The student starts from the same clean DLF checkpoint as the frozen teacher and
adds the standard missing-modality wrapper. Text is always present; each Train
sample receives one missing mode `LA`, `LV`, or `L` from the dedicated RNG:

```text
seed + 104729
```

The objective is unchanged from the original CFCompat-v1 method:

```text
L_total = L_full + L_missing + L_KD

L_KD = sum_i compat_i * SmoothL1(student_missing_i, teacher_LAV_i)
       / (sum_i compat_i + 1e-8)
```

There is no reliability gate, threshold, temperature, regret rule, projection,
v13 safety layer, or new MOSEI-specific tunable parameter in this stage.

## Validation-only checkpoint selection

Every epoch evaluates only official MOSEI Valid under `LAV`, `LA`, `LV`, and
`L`. The sole checkpoint/scheduler criterion is:

```text
J_valid = 0.5 * MAE_LAV
        + 0.5 * mean(MAE_LA, MAE_LV, MAE_L)
```

Formal training keeps the current MOSEI execution protocol:

```text
batch_size       = 16
update_epochs    = 10
learning_rate    = 1e-4
patience         = 5
early_stop       = 10
num_workers      = 0 on Windows
matmul_precision = high by default
```

No Test-best checkpoint is created.

## 8 GiB Windows execution adaptation

The historical trainer built `L_full`, `L_missing`, and `L_KD` graphs and then
called one backward on their sum. The MOSEI Windows port keeps the forward/RNG
order but executes:

```text
backward(L_full)
backward(L_missing + L_KD)
# no optimizer.step occurs between these calls
```

The optimizer step remains at the same gradient-accumulation boundary. Therefore
its accumulated mathematical gradient is:

```text
grad(L_full) + grad(L_missing + L_KD)
= grad(L_full + L_missing + L_KD)
```

This lowers peak activation memory. It is an execution adaptation, not a new
method component; floating-point summation need not be bitwise identical to the
historical single-backward implementation.

## Formal outputs

Validation-best CFCompat member checkpoints:

```text
pt/missing_baseline/cf_compat_kd_v1/mosei/seedSEED/
    DLF_mosei_seedSEED_best_valid.pth
```

Per-seed run manifests and Valid-only audit artifacts:

```text
result/missing_baseline/mosei_cfcompat_v1/seedSEED/
```

Canonical five-seed metrics:

```text
result/missing_baseline/cf_compat_kd_v1/mosei_per_seed.csv
result/missing_baseline/cf_compat_kd_v1/mosei_summary.csv
```

Each formal validation-best member also writes the exact Valid prediction format
expected by the later Raw5 code:

```text
result/missing_baseline/cfcompat_prediction_ensemble_v1/mosei/
    online_seedSEED_valid_predictions.csv
```

No Test prediction file is produced at this stage.

## Commands

Preflight:

```powershell
.\scripts\check_windows_mosei_cfcompat_v1.ps1
```

Seed1111 isolated smoke cache + two-epoch training smoke:

```powershell
.\scripts\run_windows_mosei_cfcompat_v1.ps1 -Action Cache -Seed 1111 -Smoke
.\scripts\run_windows_mosei_cfcompat_v1.ps1 -Action Train -Seed 1111 -Smoke
```

After smoke acceptance, build a formal cache then train the matching formal
member, one seed/process/GPU at a time:

```powershell
.\scripts\run_windows_mosei_cfcompat_v1.ps1 -Action Cache -Seed SEED
.\scripts\run_windows_mosei_cfcompat_v1.ps1 -Action Train -Seed SEED
```

Do not run official Test during this stage. After all five members are frozen,
Raw5 Valid is formed by equal `0.2` prediction averaging across seeds
`1111..1115`; model parameters are never averaged.
