# Stage 19 Failure Diagnosis

## Conclusion

`STAGE19D_MGD_SCREEN_FAILED` and `STAGE19_ROUTE_CLOSED`.

MGD passed the implementation and Screen-1 mechanism checks but failed the
pre-registered Screen-2 metric gate.  MGRD was not run because its parent MGD
did not promote; recoverability headroom being present does not override the
parent-method gate.

## 1. Implementation failure

Not supported.  Seventeen tests passed; threshold decisions matched a dense
project-evaluator grid; cache sample-order binding and real-DLF resume were
exact; all MGD losses were finite and non-zero.  Locked Test access remained 0.

## 2. Optimization failure

The ordinal objective was active and produced gradients, and runtime did not
exceed Uniform by 35%.  The trajectory was unstable relative to Uniform:
Screen-1 was positive, but epochs 7 and 8 were both worse.  This is consistent
with an objective/optimization mismatch rather than a dead loss.

## 3. Mechanism failure

Recoverability headroom exists (aggregate coarse-only 33.38%, c CV 0.633), but
that only licenses an MGRD test after MGD succeeds.  The ungated multi-granular
decomposition itself failed to turn its active ordinal signal into a stable
validation gain, so MGRD and shuffled controls were correctly skipped.

## 4. Metric trade-off

At Screen-2, MissingMacro MAE worsened by 0.00839.  MissingMacro Acc5 and Acc7
worsened by 0.00303 and 0.00445.  Tiny isolated Acc2/Corr changes did not
compensate for broad regression and fine-grained classification degradation.

## 5. Generalization/trajectory failure

The direction changed from Screen-1 ΔJ=-0.01512 to Screen-2 ΔJ=+0.00813.
This is a within-seed checkpoint-instability failure.  Seed1114 was not run, so
no cross-seed claim is made.

## Next step

Do not tune tau, alpha, granularity weights, learning rate, or seed under this
registration.  A future independent stage may perform a label-free gradient
conflict audit of ordinal versus supervised objectives, but it must register a
new hypothesis and baseline protocol before implementation.
