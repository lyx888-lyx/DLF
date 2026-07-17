# Frozen evaluator decision rules for Stage 9B

## Audited implementation

Stage 9A evaluates saved predictions through
`trains/singleTask/cfcompat_prediction_ensemble_utils.py::metrics_from_predictions`.
That function converts every prediction and label column to a `torch.float32`
tensor and calls
`trains/singleTask/missing_utils.py::regression_metrics`.

The older `trains/utils/metricsTop.py` implementation was also inspected. It has
the same prediction decisions, but Stage 9A uses the unrounded metric values from
`missing_utils.regression_metrics`; therefore that is the frozen authoritative
path for this experiment.

## Exact prediction transformations

All saved scalar predictions are first represented as IEEE-754 `float32`.

| Metric | Prediction decision |
|---|---|
| Acc7 | `np.round(np.clip(prediction, -3.0, 3.0))` |
| Acc5 | `np.round(np.clip(prediction, -2.0, 2.0))` |
| Acc2 | `prediction > 0` |
| F1 | the same `prediction > 0` class used by Acc2 |

`np.round` uses round-half-to-even, not Python's formatting rules and not
round-half-away-from-zero. Consequently exact half-integers are assigned to the
even class. Examples: `-2.5 -> -2`, `-1.5 -> -2`, `-0.5 -> 0`, `0.5 -> 0`,
`1.5 -> 2`, and `2.5 -> 2`.

Acc7 is clipped to `[-3, 3]` before rounding. Acc5 is clipped to `[-2, 2]`
before rounding. Acc2 has no clipping. At exactly zero, `prediction > 0` is
false.

The finite multiclass boundaries are `-2.5`, `-1.5`, `-0.5`, `0.5`, `1.5`,
and `2.5` for Acc7, and `-1.5`, `-0.5`, `0.5`, and `1.5` for Acc5. Boundary
membership follows half-to-even: even-class intervals include their finite
half-integer boundaries; odd-class intervals exclude them.

## Has-zero versus non-zero binary protocol

The legacy evaluator computes both a has-zero binary route (`prediction >= 0`)
and a non-zero route (`prediction > 0`). Stage 9A's actual reported `acc_2` and
`F1_score` use the non-zero-label route:

1. samples whose ground-truth label equals zero are excluded from these two
   aggregate metrics;
2. the prediction class for remaining samples is `prediction > 0`;
3. F1 is weighted F1 with `zero_division=0`.

ADPEP never sees labels during projection. It preserves `prediction > 0` for
every sample, including samples that will later be excluded by the evaluator.
This guarantees exact Acc2 and F1 inheritance after labels are loaded.

## Stage 9B wrapper

`trains/singleTask/anchor_decision_projection.py::evaluator_decisions` is the
label-free decision wrapper. Automated tests compare it directly with the
frozen Stage 9A metric implementation at every boundary, the adjacent float32
values, positive and negative zero, clipped extremes, and ordinary values.

Open projection boundaries are represented by the adjacent interior float32
value from `np.nextafter`. Every projected value is passed through
`evaluator_decisions` again before it can be frozen.
