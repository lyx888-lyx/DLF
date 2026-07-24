# Stage 23A-MOSI — Frozen-Protocol Parallel Transfer Audit

Final status: `STAGE23A_MOSI_PARALLEL_BLOCKED_NO_SAFE_RESOURCE`

The audit stopped at the mandatory parallel-resource gate. No Expert complementarity or Judge conclusion is permitted from partial OOF output.

## Required answers

1. MOSI speed probe was 13.035 s/epoch; the mean pre-launch MOSEI baseline was 354.752 s/epoch, so the raw dataset/model speed ratio was 27.22x.
2. Parallel execution affected MOSEI throughput: fold0 slowed 70.9% and fold1 slowed 26.3%; MOSI alone was stopped.
3. Five-fold MOSI OOF quality: not evaluable; the resource gate stopped the run before all folds/components completed.
4. Method-vs-seed complementarity: not evaluable.
5. Oracle expert selection vs per-mode fixed stacking: not evaluable.
6. Unique expert marginal contributions: not evaluable.
7. Judge error/regret prediction: Judge was not trained.
8. Joint-risk Teacher vs fixed stacking: not evaluated.
9. MOSI/MOSEI trend consistency: not evaluable from partial OOF.
10. Worth waiting for primary MOSEI Stage23A: Yes.
11. Recommend Student training: No.
12. Official Valid accessed: No.
13. Locked Test access count: 0.

## Execution facts

- GPU used: 2
- MOSI workers: one training worker; OMP_NUM_THREADS=2; DataLoader workers=2
- 3-epoch probe: 13.035 s/epoch, peak 2.366 GiB
- MOSI process group stopped: Yes
- MOSEI processes killed/paused/reniced: No
- Official Valid access count: 0
- Locked Test access count: 0
- Student trained: No
- Dependencies upgraded: No
- Existing MOSI/MOSEI worktrees modified: No
