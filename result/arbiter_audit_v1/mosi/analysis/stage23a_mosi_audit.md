# Stage 23A-MOSI — Frozen-Protocol Parallel Transfer Audit

Final status: `STAGE23A_MOSI_JUDGE_NOT_ACTIONABLE`

## Required answers

1. MOSI observed seconds/epoch: 13.035; it was 25.9x–28.5x faster per epoch than the two pre-parallel MOSEI baselines.
2. MOSEI throughput impact: the initial monitor measured fold0 +70.9% and fold1 +26.3% seconds/epoch. The automatic stop verdict was `BLOCKED_CROSS_FOLD_CORROBORATED_MOSEI_SLOWDOWN_GT_15_PERCENT`; the user explicitly waived only that runtime stop and requested continued parallel execution. MOSEI was never stopped or modified.
3. Five-fold OOF expert quality: see `oof_expert_metrics_by_fold.csv`; all predictions are held-out-source OOF.
4. Larger complementarity source: `seed_difference`.
5. Oracle ΔJ vs per-mode fixed stacking: -0.287448; improving outer folds: 5/5.
6. Retained experts with cross-fold contribution: uniform_kd_seed1111, moddrop_seed1111, moddrop_seed1114, cfcompat_seed1111, cfcompat_seed1114.
7. Judge error/regret prediction: mean Spearman 0.1967, AUROC 0.6299, ranking accuracy 0.1240.
8. Joint-risk vs fixed stacking: mean ΔJ -0.000027.
9. MOSI vs MOSEI trend: MOSEI is still running; no final cross-dataset claim is made.
10. Worth waiting for MOSEI: Yes; MOSI is only a secondary fast falsification check.
11. Recommend Student training: No.
12. Official Valid accessed: No.
13. Locked Test access count: 0.

## Runtime

- Five-fold wall time: 9982.966 seconds (2.77 hours)
- GPU: 2
- Maximum simultaneous MOSI training workers: 1
- Parallel slowdown waiver recorded: True

## Locks

- Student trained: No
- Official Valid access count: 0
- Locked Test access count: 0
- Existing MOSI/MOSEI worktrees modified: No
- Dependencies upgraded: No
