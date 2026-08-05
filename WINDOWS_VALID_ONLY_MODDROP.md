# Windows Valid-only ModDrop reconstruction

This stage consumes the reconstructed clean DLF checkpoints for seeds 1112,
1113, and 1115 and creates the validation-best ModDrop evaluators required by
the train-only counterfactual-compatibility cache.

The runner constructs exactly `train` and `valid`. It does not construct or
read the official Test split.

## Seed 1112 smoke run

```powershell
.\scripts\run_windows_valid_only_moddrop.ps1 -Seed 1112 -Overwrite
```

The smoke run is limited to two epochs and writes under `smoke/` paths. It does
not overwrite formal assets.

## Formal runs

```powershell
.\scripts\run_windows_valid_only_moddrop.ps1 -Seed 1112 -Formal -Overwrite
.\scripts\run_windows_valid_only_moddrop.ps1 -Seed 1113 -Formal -Overwrite
.\scripts\run_windowws_valid_only_moddrop.ps1 -Seed 1115 -Formal -Overwrite
```

Run one seed at a time on the 8 GB laptop GPU.

Formal per-seed CSVs are written to:

```text
result/missing_baseline/moddrop_benchmark_multiseed_v1/seedSEED/mosi_per_seed.csv
```

Formal evaluator checkpoints are written to:

```text
pt/missing_baseline/moddrop_benchmark_multiseed_v1/seedSEED/DLF_mosi_seedSEED_best_valid.pth
```

The CSV records `MainCheckpoint` and `BestValidEpoch`, which are the fields used
by the later compatibility-cache builder. Each output directory also contains
a SHA-bound manifest and validation predictions.
