# Regret-Aware Best-So-Far Memory CFCompatKD v4.2

## Motivation

Seed1113 v4 reduced negative transfer and improved Valid-J, but under-used the `better_and_correct` Teacher subgroup. v4.1 recovered that Teacher benefit but gave back a substantial fraction of v4's negative-transfer protection. The remaining failure is temporal: neither a frozen baseline nor a pause state remembers that the Student itself may already have reached a prediction better than both.

v4.2 therefore introduces a per-Train-sample/per-missing-mode best-so-far prediction memory.

## Memory

For every Train `(sample_index, missing_mode)` pair, memory is initialized from the frozen validation-best ModDrop missing-modality prediction.

Before computing the auxiliary losses for an event:

1. evaluate the detached current Student prediction;
2. compare its absolute Train-label error with the stored memory error;
3. if the current Student strictly improves the stored error, replace the memory prediction with the current detached prediction;
4. route the event against the updated memory.

Memory updates use Train labels only. Memory is not used at inference and adds no inference parameters.

## Routing

Let `m` be updated best-so-far memory, `s` the detached current Student, `t` the frozen full-modality Teacher, `y` the Train label, and `c` the frozen CFCompat score.

Teacher targets remain current-Student safe:

`teacher_safe = clip(t, min(s,y), max(s,y))`.

Preservation targets are also safe:

`memory_safe = clip(m, min(s,y), max(s,y))`.

The exclusive routes are:

- **Strong Distill**: Teacher error improves memory by at least `0.02`, and the safe Teacher target is active.
- **Weak Distill**: Teacher improves memory by `(0, 0.02)`, moves in the correct memory-to-label direction, and the safe Teacher target is active. Weak KD has true `0.25x` strength.
- **Preserve**: Teacher does not improve memory and current Student error is at least `0.02` worse than memory. Apply preservation toward `memory_safe` with coefficient `0.25`.
- **Abstain**: otherwise.

CFCompat remains only a mild distillation prior: `0.75 + 0.25*c`.

## True weak-KD normalization

v4.2 reuses the v4.1 two-tier reduction. The weak `0.25` factor is applied only in the numerator while the denominator uses unscaled eligible CFCompat mass, so an all-weak batch remains genuinely one quarter strength.

## Frozen Seed1113 development gate

Before the v4.2 run, the following criteria are frozen:

- Valid-J no more than `0.002` worse than v4;
- `better_and_correct` gain no more than `0.010` below v4.1;
- Q1/Q4 gain no more than `0.005` below v4;
- negative-transfer rate no more than `0.010` above v4;
- severe-negative-transfer rate no more than `0.010` above v4;
- positive-transfer rate no more than `0.010` below v4;
- harmful-imitation rate no more than `0.010` above v4;
- at least two non-inferior epochs versus v4;
- strong and weak distillation, preservation, and memory updates must all be exercised.

Passing this Seed1113 development gate only promotes v4.2 to the frozen 1112/1113/1115 Valid screen. It does **not** authorize official Test.

## Windows commands

Smoke:

```powershell
.\scripts\run_windows_cfcompat_regret_best_memory_valid_screen.ps1 -Overwrite
```

Formal single-seed development trajectory:

```powershell
.\scripts\run_windows_cfcompat_regret_best_memory_valid_screen.ps1 -Formal -Overwrite
```

Audit only:

```powershell
.\scripts\audit_windows_cfcompat_regret_best_memory_valid_screen.ps1
```
