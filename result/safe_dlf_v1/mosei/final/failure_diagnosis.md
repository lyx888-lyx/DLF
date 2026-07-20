# Stage 20 failure diagnosis

## Implementation

- The first long-run attempt completed epoch 1 computation but failed before resumable persistence because optional screen keys were absent.
- The failed attempt was preserved; fix commit: `c1f686e8164f729e0e31d6ad18cec21d111b0c3d`.
- The clean rerun passed all implementation gates and completed normally.

## Mechanism

- Ghost headroom was present: maximum GAR `1.000962`.
- Frozen-wrapper raw filler sensitivity was `0.000e+00`.
- Baseline unsupported gradient share was `0.175387`.
- SAFE Full removed filler sensitivity and absent gradients in the implementation gate.

## Metrics

- SAO-PM delta J: `+0.005840`; passed: `False`.
- SAFE Full delta J: `+0.003276`; passed: `False`.
- SAFE Full minus SAO-PM delta J: `-0.002565`.

No coefficient, learning-rate, batch-size, AMP, KD, or Test-based rescue was attempted.
