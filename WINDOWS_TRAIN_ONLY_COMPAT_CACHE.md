# Windows train-only compatibility-cache reconstruction

This stage consumes the validation-best ModDrop evaluator for one of the fixed
formal seeds: 1112, 1113, or 1115.

It constructs exactly one non-shuffled `train` loader. A runtime guard aborts if
any code path requests `valid` or `test`. Each cache contains exactly 1284 unique
MOSI training samples with sample indices `0..1283`.

## Update the branch

```powershell
git fetch origin
git reset --hard origin/windows/valid-only-prereq-rebuild-v1
git rev-parse HEAD
```

## Run one seed

```powershell
.\scripts\run_windows_train_only_compat_cache.ps1 -Seed 1112 -Overwrite
```

After Seed 1112 succeeds, run 1113 and 1115 one at a time.

## Formal output

```text
result/counterfactual_compatibility/cf_compat_v1_multiseed/mosi/seedSEED/
  train_counterfactual_compatibility.csv
  cf_compat_config.json
  cf_compat_summary.json
  cf_compat_mode_bins.csv
  cache_run.csv
  manifest.json
```

The config and manifest bind the cache to the exact ModDrop evaluator SHA256,
best validation epoch, Git commit, environment, and train-only loader audit.
The official Test split is not constructed or read.
