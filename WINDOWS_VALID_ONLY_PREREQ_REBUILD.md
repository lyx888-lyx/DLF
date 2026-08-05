# Windows Valid-only prerequisite reconstruction

This branch adds a Windows-native reconstruction path for the prerequisites
required by `feature/cfcompat-safe-projection-valid-screen-v1`.

## Current scope

Implemented now:

- CUDA, MOSI and local BERT preflight;
- clean DLF training for seeds `1112`, `1113`, and `1115`;
- Train/Valid-only checkpoint selection;
- Seed-1112 two-epoch smoke mode;
- checkpoint reload audit and SHA256 manifest;
- no official Test loader construction or traversal.

Not implemented yet:

- paired ModDrop evaluator reconstruction;
- train-only counterfactual compatibility cache;
- original CFCompatKD Valid-only reference reconstruction;
- final Safe-CFCompatKD Windows wrapper.

Those actions should be added only after the clean DLF smoke run succeeds.

## First run

Activate the `DLF` Conda environment and run from PowerShell:

```powershell
git fetch origin
git switch windows/valid-only-prereq-rebuild-v1

.\scripts\run_windows_valid_only_prereq.ps1 -Seed 1112 -Overwrite
```

The default command runs a two-epoch smoke test. It writes to isolated paths:

```text
pt/windows_valid_only_prereq_v1/smoke/clean_dlf/seed1112/
result/windows_valid_only_prereq_v1/smoke/clean_dlf/seed1112/
```

It does not overwrite the formal checkpoint path.

## Formal clean DLF run

After reviewing the smoke output:

```powershell
.\scripts\run_windows_valid_only_prereq.ps1 -Seed 1112 -Formal
```

The formal checkpoint is written to the path expected by later stages:

```text
pt/DLF_mosi_seed1112_best.pth
```

Repeat for `1113` and `1115` only after Seed 1112 completes successfully.

## Protocol details

The reconstruction preserves:

- the original clean DLF combined objective;
- Adam, configured learning rate, gradient clipping and early stopping;
- validation output-head L1 loss as the checkpoint criterion;
- the original behavior of dropping a partial final gradient-accumulation window.

The runner records the Python, PyTorch, CUDA, GPU, Git commit, checkpoint SHA256,
epoch trajectory, and an explicit `official_test_constructed=false` marker.
