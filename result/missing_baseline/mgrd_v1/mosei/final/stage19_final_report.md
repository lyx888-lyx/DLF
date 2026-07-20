# Stage 19B–19F MOSEI Multi-Granular Recoverable Distillation v1

## Final status

- `STAGE19B_AMP_REJECTED_FALLBACK_FP32`
- `STAGE19D_MGD_SCREEN_FAILED`
- `STAGE19_ROUTE_CLOSED`

Final retained innovation: **none**.  Uniform KD remains the baseline.  MGD
failed the pre-registered Screen-2 gate; therefore MGRD, shuffled gate, and
seed1114 were not run.

## Protocol and safety

- Base commit: `dc8536f338c7e310a73447e4185c38be52800f48`
- Implementation commit: `82934af61f59c6c81ed5109087920a443a218f62`
- Frozen batch/accumulation: 16 / 10, gradient-sum semantics preserved
- Tau: 0.5; boundaries: `{"Acc2": [0.0], "Acc5": [-1.5, -0.5, 0.5, 1.5], "Acc7": [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]}`
- Locked Test access count: **0**
- No dependency upgrades; original MOSEI worktree unchanged

## Stage 19B

BF16 was unsupported.  FP16+GradScaler was rejected.  Best-valid
ΔJ(FP16−FP32)=+0.007755.  Complete training wall was
1.871h versus 0.865h,
so FP16 was 116.2% slower because its changed
trajectory ran 28 epochs versus 12.  Multiple MAE/Corr/Acc5/Acc7 safety gates
also failed; epoch1 contained a non-finite pre-clip gradient norm.  The final
protocol therefore uses FP32.

Teacher and ModDrop predictions were cached for train/valid with ordered
sample-ID SHA binding.  Online Teacher/reference forward counts during training
were zero.  Stage19A measured the eliminated Teacher forward at 40.21ms per
microbatch, approximately 41.05s per 1021-batch epoch.

Hinge vectorization passed: GPU loss difference
5.960e-08, gradient max difference
4.657e-10.

Resume integrity passed with maximum numeric difference 0 in the real-DLF
continuous-vs-restart probe.  All 17 tests passed.

## Uniform KD FP32 seed1111

- Best epoch: 2
- J: 0.515493
- Wall: 0.865h
- LAV: Acc7=0.545697, Acc5=0.560663,
  Acc2=0.849791, F1=0.846929,
  Corr=0.750748, MAE=0.514148
- MissingMacro: Acc7=0.546945,
  Acc5=0.561554,
  Acc2=0.850487,
  F1=0.847464,
  Corr=0.747910,
  MAE=0.516839

## MGD screen

Screen-1 epoch4 was positive: ΔJ=-0.015117, all three missing
mode MAEs improved, MissingMacro MAE=-0.012819.
The method was correctly allowed to continue without restart.

Screen-2 epoch8 failed:

- ΔJ=+0.008126
- LA/LV/L MAE improved: 0/3
- MissingMacro MAE=+0.008392
- MissingMacro Corr=-0.000111
- MissingMacro Acc5=-0.003029
- MissingMacro Acc7=-0.004454
- Epoch7 ΔJ=+0.012620
- Fast-screen wall: 0.543h

MGD was not resumed to full training.  Its screen checkpoint, logs, formulas,
paired tables, and stop reason are retained.

## MGRD headroom and gate

Headroom existed: aggregate coarse-only=33.38%,
c CV=0.633, and all three missing modes exceeded the
5% mode threshold.  This does not rescue a failed parent MGD.  Consequently:

- MGRD vs MGD: **not run / not applicable**
- MGRD vs shuffled gate: **not run / not applicable**
- benefited modes/granularities: no promoted method, so no benefit claim

The result cannot be attributed only to directly optimizing Acc2/Acc5/Acc7:
the ordinal losses were active, yet MAE and fine classification broadly
degraded at Screen-2.  This is a metric/trajectory trade-off, not success.

## Seed decision

Seed1111 did not pass the MGD screen.  Seed1114 was therefore not run.  No
seed-driven generalization claim is made.

## Failure

Implementation and cache/resume integrity passed.  The failure is a
validation-trajectory/metric failure: Screen-1 improvement reversed by
Screen-2, with broad missing-mode MAE degradation and Acc5/Acc7 trade-offs.
No tau, alpha, weight, learning-rate, or seed rescue was attempted.
