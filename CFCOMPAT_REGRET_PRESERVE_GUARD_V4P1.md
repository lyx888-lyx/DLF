# Regret-Aware Preserve-or-Distill CFCompatKD v4.1

## Motivation

The Seed-1113 v4 development run improved overall Valid-J and reduced missing-modality negative transfer, but failed one frozen subgroup criterion: `better_and_correct` gain fell from the v2 Uniform reference `0.088598` to `0.058568`.  v4.1 is a targeted post-v4 repair.  It does **not** redesign the successful preservation path and it does **not** reintroduce v3 utility/difficulty weighting.

Official Test remains forbidden.

## Fixed development scope

- Dataset: MOSI.
- Development seed: `1113` only.
- New trajectories trained: exactly one, `regret_preserve_guard_cfcompat`.
- Frozen references: audited v2 `cfcompat_replay`, audited v2 `student_safe_uniform`, and independently audited v4 `regret_preserve_cfcompat`.
- Checkpoint selection: minimum official Valid-J.
- Same optimizer, update cadence, missing-mode RNG sequence, Teacher, ModDrop baseline, compatibility cache, and Train/Valid-only protocol as the reconstructed Stage-3 chain.

## Frozen signals

For one Train sample/missing-mode event:

- `b`: frozen validation-best ModDrop missing-modality prediction.
- `s`: detached current missing-modality Student prediction.
- `t`: frozen full-modality Teacher prediction.
- `y`: Train label.
- `c`: frozen train-only CFCompat compatibility.

Define

```text
baseline_error = |b-y|
teacher_error  = |t-y|
current_error  = |s-y|
teacher_advantage = baseline_error - teacher_error
current_regret     = current_error - baseline_error
```

The Teacher target is still clipped to the closed current-Student-to-label interval.  A Teacher that the current Student has already matched or surpassed is therefore not allowed to pull the Student backward.

## Five explicit routing states

### 1. STRONG_DISTILL

```text
teacher_advantage >= 0.02
AND current-Student-safe Teacher target is active
```

Use normal mild-CFCompat KD.

### 2. WEAK_DISTILL

```text
0 < teacher_advantage < 0.02
AND Teacher moves in the correct frozen-baseline-to-label direction
AND current-Student-safe Teacher target is active
```

Use the same safe Teacher target, but its KD contribution is fixed to `0.25x` strong strength.

### 3. BENEFICIAL_PAUSE

```text
teacher_advantage > 0
AND current-Student-safe Teacher target is inactive
```

The Student has already matched/surpassed this beneficial Teacher at the current step.  KD is paused, not reversed.  Routing is recomputed every batch, so the Teacher automatically re-enters strong/weak distillation if the Student later regresses behind it.

### 4. PRESERVE

```text
teacher_advantage <= 0
AND current_regret >= 0.02
```

Teacher is not genuinely better than the frozen baseline and the current Student has materially regressed.  Apply the unchanged v4 preservation loss toward the frozen ModDrop prediction clipped to the current-Student-to-label interval.

### 5. ABSTAIN

All remaining events receive neither KD nor preservation.

The five states are mutually exclusive and exhaustive.

## CFCompat role

For strong/weak eligible distillation only:

```text
mild_cfcompat = 0.75 + 0.25 * compatibility
```

CFCompat remains a prior, not the safety decision-maker.

## Important weak-KD normalization fix

The repository's original `gated_kd_loss` computes

```text
sum(gate * loss) / sum(gate)
```

Therefore replacing `gate` by `0.25 * gate` does **not** make an all-weak batch 0.25x: the factor cancels in numerator and denominator.

v4.1 instead defines

```text
eligible_mass = strong_gate + weak_gate
effective_gate = strong_gate + 0.25 * weak_gate

L_KD = sum(effective_gate * SmoothL1(s, t_safe))
       / (sum(eligible_mass) + eps)
```

Consequences:

- all-strong batches exactly retain v4 normalization;
- all-weak batches are genuinely `0.25x`;
- mixed batches preserve the intended strong/weak contribution ratio.

## Total training objective

```text
L_total = L_full_supervised
        + L_missing_supervised
        + L_tiered_KD
        + 0.25 * L_preserve
```

The v4 preserve margin `0.02` and coefficient `0.25` are unchanged.

## Frozen post-v4 Seed-1113 development checks

This is explicitly a targeted post-v4 development screen, not a preregistered formal claim.  Before running v4.1, the following criteria are frozen:

- Candidate Valid-J no worse than v4 by more than `0.002`.
- `better_and_correct` gain improves over v4 by at least `0.010`.
- Q1 and Q4 gains each no worse than v4 by more than `0.005`.
- Missing-modality negative-transfer rate no worse than v4 by more than `0.010`.
- Missing-modality positive-transfer rate no worse than v4 by more than `0.010`.
- Harmful-imitation rate no worse than v4 by more than `0.010`.
- At least two epochs are noninferior to `v4_J + 0.002`.
- Strong and weak distillation are both exercised; preservation and a non-distill path are exercised.

A pass only promotes the frozen mechanism to the existing 1112/1113/1115 Valid screen.  It does not authorize official Test.

## Windows commands

Smoke:

```powershell
.\scripts\run_windows_cfcompat_regret_preserve_guard_valid_screen.ps1 -Overwrite
```

Formal Seed-1113 development trajectory:

```powershell
.\scripts\run_windows_cfcompat_regret_preserve_guard_valid_screen.ps1 -Formal -Overwrite
```

Audit only:

```powershell
.\scripts\audit_windows_cfcompat_regret_preserve_guard_valid_screen.ps1
```
