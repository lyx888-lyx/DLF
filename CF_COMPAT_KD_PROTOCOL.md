# Counterfactual Compatibility-Gated Prediction KD (Stage 3B)

This implementation registers two fixed methods: DLF-CFCompatKD-v1 (compat)
and DLF-ReliabilityCFCompatKD-v1 (reliability_compat). Both use the
benchmark/original-protocol track: validation and test are evaluated each epoch,
but the main checkpoint is selected only by validation J. The lowest-test
checkpoint is written solely under an isolated diagnostic path.

The full teacher is a frozen plain Gate 3 DLF LAV model. The trainable student
is an independently initialized MissingModalityWrapper(DLF) with the unchanged
Stage 1 mask tokens and adapter. The counterfactual evaluator is a separate,
frozen Stage 1 ModDrop validation-best model. Its checkpoint is read from the
Stage 1 per-seed result CSV, not inferred from a convention.

Before training, the evaluator makes exactly one non-shuffled, train-only pass
over all samples to cache E_LAV/E_LA/E_LV/E_L. Per missing mode:
delta = abs(E_LAV - E_mode), stable average ranks are computed using NumPy
mergesort, q = (rank - 0.5)/N, and compatibility = 1 - q. Cache construction
preserves Python, NumPy, torch CPU/CUDA, and the dedicated missing-mask
generator state.

For a sampled LA/LV/L view, C-only uses w=compatibility and R-times-C uses
w=exp(-abs(teacher_LAV-label))*compatibility. Gates are detached and affect
only output-logit SmoothL1 KD:
sum(w_i * kd_i) / (sum(w_i) + 1e-8). Full DLF and missing task losses remain
unweighted. Eta and lambda-KD are both fixed at 1.0.

The cache, its hash-bearing configuration, summary, and bins reside under
result/counterfactual_compatibility/cf_compat_v1/mosi/. Each method produces
per-seed/summary/epoch metrics, real gate summary and quartile diagnostics,
valid-best predictions, and separately labelled diagnostic-test predictions.
