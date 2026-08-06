# Student-Safe Dynamic-Utility Residual-CFCompat v3

This branch is an exploratory Valid-only successor to:

- `feature/cfcompat-safe-projection-valid-screen-v1`
- `feature/cfcompat-student-safe-abstain-valid-screen-v2`

It does not overwrite either result tree and does not authorize official Test.

## Research question

The v2 screen showed that current-Student interval projection plus unsafe-KD
abstention improved mean Valid J and reduced mean harmful imitation, while
multiplying active samples directly by the frozen CFCompat score consistently
hurt all three seeds.  v3 tests whether the two ideas can be combined by making
current dynamic usefulness primary and retaining CFCompat only as a bounded
residual prior.

## Frozen runs

Formal seeds remain `1112, 1113, 1115`.

1. `cfcompat_replay`
2. `student_safe_uniform`
3. `student_safe_utility`
4. `student_safe_utility_residual_cfcompat`

All runs start from the same clean DLF initialization and use the same frozen
Teacher, missing-mode sequence, optimizer, early-stop rule, Train split and
official Valid split.

## Candidate target and gates

For a training event:

- `s`: detached current missing-modality Student prediction
- `t`: frozen full-modality Teacher prediction
- `y`: training label
- `c`: frozen train-only CFCompat score

The safe target is:

```text
t_safe = clip(t, min(s, y), max(s, y))
```

The active mask is:

```text
active = 1 when t_safe != s, otherwise 0
```

Thus wrong-direction, equal-target and zero-width events explicitly abstain
from KD.

Dynamic utility is the fraction of current absolute error removed:

```text
utility = clamp((|s-y| - |t_safe-y|) / (|s-y| + 1e-8), 0, 1)
```

The per-mode difficulty scale `tau_mode` is frozen before candidate training
from a separate Train-only pass of the initial Student:

```text
tau_mode = median_train |initial_student_mode_prediction - y|
difficulty = |s-y| / (|s-y| + tau_mode)
```

Residual CFCompat uses the pre-frozen constant `alpha=0.5`:

```text
residual_compat = 0.5 + 0.5*c
```

The four KD gates are:

```text
cfcompat_replay:
    c

student_safe_uniform:
    active

student_safe_utility:
    active * utility * difficulty

student_safe_utility_residual_cfcompat:
    active * utility * difficulty * (0.5 + 0.5*c)
```

The gated SmoothL1 KD remains normalized by the sum of gate values, so v3
changes relative allocation across safe events rather than merely shrinking a
global KD coefficient.

## Frozen decision rules

v3 reuses the v2 non-inferiority and safety thresholds without post-result
relaxation:

- mean Valid-J degradation versus CFCompat replay at most `0.003`
- each seed Valid-J degradation at most `0.005`
- each seed/mode degradation at most `0.005`
- harmful imitation must not increase on any seed
- mean harmful-imitation reduction at least `0.02`
- Q1 easy-group gain may not degrade by more than `0.002`
- Q4 hard-group gain must retain at least `80%` when replay gain is positive
- better-and-correct gain may not degrade by more than `0.002`
- at least two non-inferior epochs per seed

The primary candidate is
`student_safe_utility_residual_cfcompat`.  A pass promotes only to the planned
MOSI `1111/1114` extension, not to Test.

## Outputs

Formal results:

```text
result/missing_baseline/cfcompat_student_safe_utility_v3/mosi/valid_screen
```

Formal checkpoints:

```text
pt/missing_baseline/cfcompat_student_safe_utility_v3/mosi/valid_screen
```

The output records:

- active and abstain fractions
- utility, difficulty and final-gate means
- gate effective sample size
- compatibility/utility Pearson and Spearman correlations on active events
- per-mode active, utility, difficulty, gate and ESS summaries
- Train-only difficulty scales
- Valid raw predictions and independently recomputable safety groups
- checkpoint, cache, evaluator and reference hashes

## Windows commands

Update and switch:

```powershell
git fetch origin `
  "refs/heads/feature/cfcompat-student-safe-utility-residual-valid-screen-v3:refs/remotes/origin/feature/cfcompat-student-safe-utility-residual-valid-screen-v3"

git switch -C feature/cfcompat-student-safe-utility-residual-valid-screen-v3 `
  origin/feature/cfcompat-student-safe-utility-residual-valid-screen-v3

git rev-parse HEAD
```

Run the real-chain Seed-1112 smoke first:

```powershell
.\scripts\run_windows_cfcompat_student_safe_utility_valid_screen.ps1 -Overwrite
```

After smoke completion, run the formal 3-seed x 4-run grid once:

```powershell
.\scripts\run_windows_cfcompat_student_safe_utility_valid_screen.ps1 -Formal -Overwrite
```

Do not open or evaluate official Test during v3 development.  Test evaluation
is deferred until the method, thresholds, five-seed extension and best-Valid
checkpoint hashes are frozen.
