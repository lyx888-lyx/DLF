# Stage 4A Deterministic Counterfactual Residual Recovery

- Base commit: `276fc26e6e7d1998f7dd746edfca1efbbb691e55`.
- Branch: `feature/cf-residual-recovery-v1`.
- Seed: 1111 only; benchmark/original protocol; corrected validation J selects the main checkpoint.
- Source cache: the locked Stage 3 seed1111 train-only compatibility cache resolved through its manifest and SHA-256.
- Signed target: `r_m = E_LAV - E_m`; `scale_m = std_population(r_m)`; `z_m = r_m / scale_m`.
- Student: Gate 3 initialized `MissingModalityWrapper(DLF)` plus three independent zero-initialized `Linear(fusion_dim, 1)` heads.
- Corrected missing prediction: `base_output_logit + scale_m * z_hat_m`; LAV correction is exactly zero.
- CFRR-only loss: `L_full + L_missing + L_residual`.
- Combined loss: `L_full + L_missing + L_direct_KD + L_residual`.
- Direct KD is SmoothL1 on the base prediction weighted by detached compatibility `C`.
- Residual loss is SmoothL1 on standardized signed residual weighted by detached `1-C`.
- `eta=lambda_kd=lambda_residual=1`; no reliability, threshold, clipping, temperature, nonlinear head, feature KD, or label reweighting.
- Teacher is frozen and train-only. The Stage 1 evaluator is never instantiated during Stage 4A training.
- Valid/test inference uses only Student, residual heads, and checkpointed scales; main and test-best diagnostic checkpoints are isolated.
- Formal order: cache audit, two smoke runs, tests, commit/push, CFRR-only, combined method, final audit, stop.
