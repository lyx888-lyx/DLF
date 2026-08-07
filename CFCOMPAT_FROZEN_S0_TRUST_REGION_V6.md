# CFCompatKD v6 — Frozen-S0 Functional Trust Region

## Research question

The v5.2 Valid-only failure analysis found that many Teacher-beneficial events were already strong under the common frozen initial Student `S0`, but later v4 shared-parameter updates moved the final Student across the frozen ModDrop baseline and produced negative transfer. Failures were strongly clip-level: LA/LV/L often failed together.

v6 tests one mechanism only:

> Does protecting the pre-training Student function on Train events where `S0` is already demonstrably better than the frozen baseline reduce beneficial-Teacher negative transfer without giving back v4's safety gains on Teacher-nonbeneficial events?

This is not another Teacher-usefulness classifier and does not tune v5 gates.

## Protocol boundary

- Dataset: MOSI.
- Development seed: `1113` only.
- New trajectories: exactly one.
- Decision split: official Valid only.
- Official Test: forbidden; never constructed or accessed.
- Base objective: v4 `DISTILL / PRESERVE / ABSTAIN` unchanged.
- Frozen reference: existing v4 Seed1113 Valid artifacts.
- Checkpoint selection: minimum official Valid `J`, identical to the existing development protocol.

## Frozen S0 anchor

Immediately after the usual Teacher/Student initialization and before any v6 optimization step, the code caches the Student's predictions for all Train and Valid samples in `LAV`, `LA`, `LV`, and `L` modes. The Train DataLoader generator state and global RNG state are preserved around this cache pass. No live second Student is retained, so the mechanism adds no persistent third DLF model in GPU memory.

For a Train missing-mode event, let `b` be the frozen ModDrop baseline, `s0` the cached initial Student, `s` the current Student, and `y` the Train label. Define `S0 advantage = |b-y| - |s0-y|`. An event is S0-protected when `S0 advantage >= 0.02`.

## One-sided safe trust target

The cached `s0` value is projected onto the closed interval between the detached current Student `s` and the Train label `y`.

- If current Student is already at least as good as S0, the projected target equals current Student and trust loss is zero.
- If current Student has regressed behind a protected S0, the cached S0 prediction becomes a safe corrective target.
- S0 overshoot beyond the label is clipped to the label.
- Non-protected events receive trust weight zero.

The new loss is active-mass-normalized Smooth-L1 with frozen coefficient `lambda_s0_trust = 1.0`.

The total objective is therefore:

`v4 full loss + v4 missing loss + v4 KD + 0.25 * v4 preserve + 1.0 * S0 trust`.

No v4 Teacher rule, compatibility prior, margin, safe projection, or preserve coefficient is changed.

## Frozen mechanism-signal checks

This branch trains one candidate only; there is no hyperparameter grid. The post-run mechanism signal compares the candidate with frozen v4:

1. Valid `J` degradation vs v4 <= `0.002`.
2. Teacher-beneficial negative-transfer rate improves by at least `5` percentage points.
3. Teacher-nonbeneficial negative-transfer rate worsens by no more than `3` percentage points.
4. Overall missing-mode negative-transfer rate worsens by no more than `1` percentage point.

Valid labels are used after training for this diagnostic decomposition only.

## Main outputs

The aggregated output directory is:

`result/missing_baseline/cfcompat_frozen_s0_trust_region_v6/mosi/valid_screen/seed1113_dev`

Important files:

- `frozen_s0_v6_valid_screen_summary.json`
- `frozen_s0_v6_transfer_summary.csv`
- `frozen_s0_v6_valid_s0_drift_events.csv`
- `frozen_s0_v6_valid_s0_drift_summary.csv`
- `frozen_s0_v6_train_decisions.csv`
- `frozen_s0_v6_train_s0_cache.csv`
- `frozen_s0_v6_valid_s0_cache.csv`
- `frozen_s0_v6_candidate_raw_valid_events.csv`
- `frozen_s0_v6_v4_reference_raw_valid_events.csv`

## Windows run

```powershell
.\scripts\run_windows_cfcompat_frozen_s0_trust_region_v6.ps1 -Overwrite
```

For an infrastructure-only two-epoch smoke run:

```powershell
.\scripts\run_windows_cfcompat_frozen_s0_trust_region_v6.ps1 -SmokeOnly -Overwrite
```

## Interpretation

First compare v6 vs v4 NTR among Teacher-beneficial events, then Teacher-nonbeneficial events, then final-vs-S0 drift and baseline-crossing on `S0_BENEFICIAL` / `S0_AND_TEACHER_BENEFICIAL`. If harmful events remain, inspect clip-level LA/LV/L co-failure again.

Do not use Test to choose or repair v6.
