# Stage 12A DSPC-A Protocol

DSPC-A is a validation-only audit. It reconstructs the frozen Stage 11A
correction with the fixed 0.10 step, then reuses the Stage 9B
`anchor_decision_projection.py` implementation to project each raw correction
into the baseline CFCompatKD Acc7/Acc5/Acc2 decision-safe interval.

Projection accepts only baseline and raw predictions. It does not accept labels,
errors, correctness, bins, or metrics. Predictions are written and SHA-frozen
before validation labels are loaded. No model is trained, no checkpoint is
modified, no prototype is updated, no OT is run, and no test loader is
constructed.

The audit stops after reporting the frozen Stage 12A evidence gate. Stage 12B
is never executed by this protocol.
