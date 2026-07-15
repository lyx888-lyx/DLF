# Stage 3A Reliability-Gated Prediction KD

`DLF-ReliabilityKD-v1` is the DLF benchmark/original-protocol track.  It is
separate from the frozen Stage 2.5 one-time milestone audit.  During training,
valid **and test** are evaluated after every epoch with `shuffle=False`; this is
not blind, clean, or test-once evaluation.  Checkpoints are selected only by
`J_valid`; test metrics and the best observed test epoch are diagnostics only.

Teacher and student independently start from the same Gate 3 clean checkpoint.
The teacher is plain DLF, LAV-only, `eval`, fully frozen, outside the optimizer,
and called under inference mode.  The student is the unchanged ModDrop wrapper.
For a train sample only, `w=exp(-abs(teacher_LAV.output_logit-label))`; no
clipping, temperature, threshold, class weighting, LDS, feature KD, or new
parameters are used.  The only changed loss is
`sum(w * SmoothL1(student_missing.output_logit, teacher_pred))/(sum(w)+1e-8)`.
The full task and auxiliary loss and the missing five-head task loss remain
unweighted.  `eta=lambda_kd=1.0` are fixed.

After early stopping, the valid-best student checkpoint is reloaded and both
splits are recomputed.  Its test result is the primary reported test outcome;
BestObservedTestEpoch is marked diagnostic-only.
