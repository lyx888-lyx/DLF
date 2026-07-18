# CS-DFMRC Decision Rules

Stage 13 reuses `anchor_decision_projection.py` without duplicating evaluator
logic. A decision cell is the exact `(Acc7, Acc5, Acc2)` signature returned by
the frozen evaluator. Its maximal connected feasible interval uses the formal
round-half-to-even, zero, open-boundary, `nextafter`, post-verification, and
fallback rules.

For residual grouping and finite interval-width diagnostics only, this interval
is intersected with the frozen MOSI output/label domain `[-3, 3]`. Predictions
are always projected through the formal Stage 9B ADPEP-All implementation.
