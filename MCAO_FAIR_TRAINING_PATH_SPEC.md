# MCAO Fair Training Path Specification

- Actual student initialization: a `MissingModalityWrapper` whose DLF backbone is loaded from the matched clean seed checkpoint.
- The CFCompatKD path does not initialize from a separately trained ModDrop checkpoint; no ModDrop retraining may be added.
- Original, MCAO-PM, and MCAO-AN must use the same clean teacher/student initial state, locked compatibility cache, missing sequence, data order, RNG, optimizer, scheduler, accumulation, clipping, and epoch rule.
- Stage17 must replace the historical per-epoch valid/test loop with the preregistered video-group inner split and a single official-valid evaluation.
- Official valid and test cannot select PM versus AN. The formal variant is frozen from Stage17A train-only expected-contribution imbalance.
