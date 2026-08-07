# CFCompatKD v5.2 — Student-Transfer Failure Diagnostic

## Research question

v5.1 showed that Train→Valid Teacher-usefulness relations can shift or flip. v5.2 asks the next question:

> Even when the frozen full-modality Teacher is genuinely beneficial on a Valid sample × missing-mode event, does the validation-best Student actually realize that benefit?

This separates two mechanisms that must not be conflated:

1. **selection failure** — a gate predicts the wrong Teacher-usefulness state;
2. **transfer/optimization failure** — the Teacher is genuinely useful, yet the Student does not improve or actively regresses.

## Protocol boundary

- Development seed: **1113** only.
- Split: **official Valid only**.
- Student training: **none**.
- Checkpoint selection: **none**; inputs are already-frozen validation-best artifacts.
- Test: **forbidden and never constructed/read**.
- v5 gate probabilities are joined only for diagnosis. v5 did not train the v4 Student.

Historical v4 did not preserve per-epoch Valid predictions and did not have a per-Valid-sample training route (DISTILL/PRESERVE/ABSTAIN routing occurred on Train events). v5.2 therefore does **not fabricate** Valid `route`, `KD_active`, or epoch-trajectory fields.

## Frozen inputs

```text
result/missing_baseline/cfcompat_regret_preserve_v4/mosi/valid_screen/seed1113_dev/regret_preserve_v4_candidate_raw_valid_events.csv
result/missing_baseline/cfcompat_regret_preserve_v4/mosi/valid_screen/seed1113_dev/regret_preserve_v4_reference_raw_valid_events.csv
result/missing_baseline/cfcompat_crossfit_transfer_risk_v5/mosi/valid_screen/seed1113_dev/gate_only/crossfit_transfer_risk_v5_valid_gate_diagnostic.csv
```

The v4 candidate file contains the frozen validation-best v4 Student. The reference file contains frozen `cfcompat_replay` and `student_safe_uniform`. The v5 gate file supplies Teacher-usefulness truth, gate probability, label-free prediction geometry, and a frozen initial-Student reference.

## Event definitions

For every Valid `sample × missing mode`:

- `teacher_advantage = baseline_error - teacher_error`
- Teacher beneficial: `teacher_advantage >= 0.02`
- `student_gain_vs_baseline = baseline_error - student_error`
- positive transfer: `student_gain_vs_baseline > 0.02`
- negative transfer: `student_error - baseline_error > 0.02`
- severe negative transfer: `student_error - baseline_error > 0.10`

Four coarse quadrants:

1. `teacher_helpful_student_improved`
2. `teacher_helpful_student_not_improved`
3. `teacher_not_helpful_student_improved`
4. `teacher_not_helpful_student_not_improved`

The key failure family is **Teacher helpful but Student not improved/regressed**. If this family is large, a better Teacher-usefulness gate alone cannot explain or solve the remaining failure.

## Outputs

Default output directory:

```text
result/missing_baseline/cfcompat_student_transfer_failure_diagnostic_v5p2/mosi/seed1113_valid
```

Generated files:

- `student_transfer_failure_table.csv` — v4 sample-level Student/Teacher/baseline/v5-gate joined table;
- `reference_run_transfer_table.csv` — original CFCompat replay and Uniform event table;
- `student_transfer_summary.csv` — positive/negative/severe transfer by run and missing mode;
- `teacher_student_quadrant_summary.csv` — four-quadrant composition;
- `v4_vs_reference_sample_deltas.csv` — exact samples where v4 improves/worsens versus replay/Uniform;
- `beneficial_teacher_failure_feature_summary.csv` — geometry differences between successful and failed beneficial-Teacher transfer;
- `beneficial_teacher_failure_rules.csv` — one/two-factor common patterns enriched among beneficial-Teacher failures;
- `joint_gate_student_failure_summary.csv` — v5 gate error type × realized v4 Student outcome;
- `top_student_failure_samples.csv` — prioritized sample IDs for manual failure inspection;
- `student_transfer_failure_summary.json` — compact scientific summary and guardrails.

## Windows run

```powershell
.\scripts\run_windows_cfcompat_student_transfer_failure_diagnostic_v5p2.ps1 -Overwrite
```

The runner compiles both Python files, executes a synthetic smoke test, verifies the three frozen input artifacts, then runs the real Valid-only analysis.

## Interpretation

### A. Failures mainly occur when Teacher is not beneficial

Teacher selection remains the primary bottleneck. Shift-aware / invariant routing is justified.

### B. Many failures occur although Teacher is beneficial

Gating alone is insufficient. Prioritize shared-parameter interference, target reachability, modality-information irrecoverability, and optimization stability.

### C. v5 gate says true-positive beneficial Teacher, but v4 Student still fails

This is especially important: even a correct Teacher-usefulness decision is not sufficient for successful transfer. Those samples become the highest-priority targets for a later gradient/trajectory diagnostic.

## Scientific caveat

`beneficial Teacher + Student regression` is evidence that Teacher selection is not the whole story, but it does **not** prove gradient interference causally. Representation limits, missing-information irrecoverability, and other shared optimization effects remain alternative explanations.
