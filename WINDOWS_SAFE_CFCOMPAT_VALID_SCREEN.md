# Windows Safe-CFCompatKD Valid-only screen

This is the final held-out-seed MOSI validation screen.  It consumes the
Windows-reconstructed clean DLF checkpoints, validation-best ModDrop
evaluators, train-only compatibility caches, and original CFCompatKD Valid-only
references for seeds 1112, 1113, and 1115.

The frozen grid is:

- `cfcompat_replay`
- `safe_uniform`
- `safe_cfcompat`

The official Test split is forbidden and is not constructed or traversed.

## Real-chain smoke

The default PowerShell mode executes seed 1112 for all three runs, with two
epochs per run.  It exercises the real Teacher, Student, ModDrop evaluator,
compatibility cache, safe projection, optimizer, and Valid evaluation.  It does
not apply the formal replay gate because a two-epoch trajectory cannot be
compared with a completed early-stopped reference.

```powershell
.\scripts\run_windows_safe_cfcompat_valid_screen.ps1 -Overwrite
```

Smoke outputs are isolated under:

```text
result/missing_baseline/cfcompat_safe_projection_v1/mosi/valid_screen/smoke
pt/missing_baseline/cfcompat_safe_projection_v1/mosi/valid_screen/smoke
```

## Formal 3-seed x 3-run screen

After the real-chain smoke passes:

```powershell
.\scripts\run_windows_safe_cfcompat_valid_screen.ps1 -Formal -Overwrite
```

The formal runner executes all nine trajectories sequentially, applies the
per-seed replay gate against the reconstructed original CFCompatKD references,
computes the preregistered candidate gates, writes the final verdict, and runs
the independent artifact audit.

Formal outputs are written to:

```text
result/missing_baseline/cfcompat_safe_projection_v1/mosi/valid_screen
pt/missing_baseline/cfcompat_safe_projection_v1/mosi/valid_screen
```

Important final files include:

```text
safe_projection_valid_grid_summary.csv
safe_projection_valid_screen_summary.json
safe_projection_valid_screen_report.md
safe_projection_source_manifest.json
```

The formal command is intentionally monolithic.  Do not run seeds or methods in
parallel and do not update the Git branch while it is running.
