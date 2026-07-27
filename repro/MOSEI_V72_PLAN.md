# MOSEI anchored complementarity V7.2 evaluation plan

Use the five existing CFCompat MOSEI best-valid checkpoints under the `DLF-mosei-generalization-v1` worktree. Run committee-only evaluation first; do not train the residual student until the teacher pool and anchored shrinkage are shown to transfer.

Recommended order:

1. Build or reuse the five-teacher MOSEI cache.
2. Fit global and softly region-conditioned simplex weights using only MOSEI Valid.
3. Select the anchor-to-committee shrinkage coefficient on MOSEI Valid using a 0.05 grid and a severe-harm penalty.
4. Evaluate Test once and report paired bootstrap confidence intervals.
5. Only run full V7.1 student distillation when the selected anchored committee clearly improves over the best single teacher and uniform ensemble.

Entrypoint: `evaluate_anchored_committee_v7_2.py`

Convenience command: `scripts/run_mosei_anchored_committee_v72.sh`
