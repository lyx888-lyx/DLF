# CFCompatKD Fair No-Test Training Protocol

## Recovered reference path

The engineering reference is
`run_cfcompat_stability_multiseed.py::train_one_seed`, specifically its Online
student trajectory. The recovered trainer retains:

- clean validation-selected DLF initialization for both Teacher and student;
- the seed-matched frozen ModDrop evaluator and train-only compatibility cache;
- a shuffled full MOSI train loader and the official validation loader only;
- `torch.Generator(seed + 104729)` for LA/LV/L sampling;
- full-view DLF loss plus one sampled missing-view task loss per train sample;
- SmoothL1 prediction KD normalized by the sum of detached gate weights;
- Adam with learning rate `1e-4`, update accumulation every 10 batches;
- no gradient clipping, matching the historical CFCompat Online path;
- `ReduceLROnPlateau(mode=min, factor=0.5, patience=5)`;
- official-valid `J` selection with a `1e-6` improvement threshold;
- early stop after 10 epochs without a new validation-best checkpoint;
- the same seed, data order, initialization order, and evaluation order.

The Test loader, Test features, Test labels, Test predictions, per-epoch Test
evaluation, diagnostic Test checkpoint, EMA, and trajectory soups are removed.

## Historical discrepancy resolved explicitly

The older paired ModDrop script clipped gradients at value 0.6, while the
specified historical CFCompat Online path did not clip. Stage 18 requires one
trainer for all methods. Therefore the frozen Stage 18 trainer uses no clipping
for every control. This preserves the requested CFCompat reference semantics
and prevents method-specific optimizer behavior. New ModDrop is a fair control,
not a claim of bitwise replay of the older clipped ModDrop checkpoint.

The historical `gated_kd_loss` divides by gate mass within each batch. A
constant positive gate is consequently equivalent to Uniform KD up to the
fixed `1e-8` denominator stabilizer. Equal-Mass remains registered because it
audits gate mass, but this mathematical equivalence must be reported rather
than interpreted as a separately tuned KD strength.

## Reproducibility and selection

All controls start from the same seed-specific student tensor state and use the
same Teacher, evaluator, train samples, missing-mode generator, batch sampler,
optimizer, scheduler, maximum epoch, early-stop rule, validation checkpoint
selection, supervised losses, KD coefficient, accumulation, and no-clipping
policy. Methods may differ only in the registered KD gate or Teacher binding.

Stage 18A passes only if both repeated methods differ by at most 0.003 in
validation J, new CFCompat mean J differs from historical seed-1114 Online by
at most 0.010, LAV MAE and MissingMacro MAE each differ by at most 0.015, the
selected epoch is reasonable, all assets bind by SHA-256, and Test access is
zero.
