# Current-Student Safe-Abstention CFCompatKD v2

This branch is an exploratory follow-up to the audited Safe-CFCompatKD v1
negative result. It does not revise or overwrite the v1 verdict.

## Mechanism

For each sampled missing-modality training example:

1. Run the current Student and detach its missing-modality prediction `s`.
2. Run the frozen full-modality Teacher to obtain `t`.
3. Let the supervised label be `y`.
4. Clip `t` to the closed interval between `s` and `y`.
5. If the clipped target equals `s`, assign KD gate zero (abstain).
6. Otherwise use either a unit active gate or active × compatibility.

The interval construction does not create a gradient path through `s`.

## Runs

- `cfcompat_replay`: unchanged original CFCompatKD replay.
- `student_safe_uniform`: Student-safe target with binary active/abstain gate.
- `student_safe_cfcompat`: Student-safe target with active × compatibility gate.

Formal seeds remain `1112`, `1113`, and `1115`. The official Test split is
forbidden and is never constructed.

## Reused assets

The v2 screen reuses the completed Windows assets for each seed:

- Clean DLF Teacher/Student initialization.
- Validation-best ModDrop evaluator and result manifest.
- Locked 1284-sample train-only compatibility cache.
- Original CFCompatKD Valid-only reference and checkpoint.

No prerequisite retraining is required.

## Frozen exploratory decision rule

The rule is frozen before v2 training, but the screen is explicitly exploratory
because the same three seeds were already examined during v1 analysis.

- Mean Valid-J degradation versus CFCompatKD must be at most `0.003`.
- Each seed Valid-J degradation must be at most `0.005`.
- No seed/mode MAE degradation may exceed `0.005`.
- Harmful imitation must not increase on any seed.
- Mean harmful-imitation reduction must be at least `0.02` absolute.
- Q1-easy gain may not degrade by more than `0.002` on any seed.
- Q4-hard must retain at least 80% of positive CFCompatKD gain on every seed.
- Better-and-correct gain may not degrade by more than `0.002` on any seed.
- Every seed must have at least two non-inferior validation epochs.

A passing main candidate advances only to the deferred `1111`/`1114`
extension, not to official Test evaluation.

## Windows commands

Update and switch to the branch:

```powershell
git fetch origin
git switch -C feature/cfcompat-student-safe-abstain-valid-screen-v2 `
  origin/feature/cfcompat-student-safe-abstain-valid-screen-v2
```

Run the Seed 1112 two-epoch real-chain smoke:

```powershell
.\scripts\run_windows_cfcompat_student_safe_abstain_valid_screen.ps1 `
  -Overwrite
```

After the smoke passes, run the formal 3-seed × 3-run grid:

```powershell
.\scripts\run_windows_cfcompat_student_safe_abstain_valid_screen.ps1 `
  -Formal `
  -Overwrite
```

## Output

Formal results:

```text
result/missing_baseline/cfcompat_student_safe_abstain_v2/mosi/valid_screen
```

Formal checkpoints:

```text
pt/missing_baseline/cfcompat_student_safe_abstain_v2/mosi/valid_screen
```

The formal PowerShell runner automatically executes the independent audit after
all nine trajectories finish.
