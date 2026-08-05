# Windows Valid-only original CFCompatKD reference

This stage consumes, for seeds 1112, 1113, and 1115:

- the validation-best clean DLF checkpoint;
- the validation-best ModDrop evaluator;
- the 1284-sample train-only compatibility cache.

It runs the exact memory-safe Safe-v3 `cfcompat_replay` trajectory and writes
the Stage-3 reference CSV required by the later Safe-CFCompatKD replay gate.
Only Train and official Valid are constructed. Official Test is forbidden.

## Seed 1112 smoke run

```powershell
.\scripts\run_windows_valid_only_cfcompat_reference.ps1 -Seed 1112 -Overwrite
```

The smoke run is limited to two epochs and writes under `smoke/` paths.

## Formal runs

```powershell
.\scripts\run_windows_valid_only_cfcompat_reference.ps1 -Seed 1112 -Formal -Overwrite
.\scripts\run_windows_valid_only_cfcompat_reference.ps1 -Seed 1113 -Formal -Overwrite
.\scripts\run_windows_valid_only_cfcompat_reference.ps1 -Seed 1115 -Formal -Overwrite
```

Run one seed at a time.

Formal reference CSVs:

```text
result/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seedSEED/mosi_per_seed.csv
```

Formal checkpoints:

```text
pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seedSEED/DLF_mosi_seedSEED_best_valid.pth
```

Each per-seed output directory also contains epoch metrics, validation
predictions, raw validation events, a summary, and a SHA-bound manifest.  The
later Safe-v3 replay gate requires exact best-epoch agreement and at most
`1e-4` difference in Valid J and each Valid modality MAE.
