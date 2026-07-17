# Stage 9B ADPEP protocol

## Scope

Anchor Decision-Preserving Ensemble Projection (ADPEP) is a deterministic,
post-inference operation over existing Stage 9A scalar predictions. It performs
no training, model inference, parameter averaging, ensemble-weight search,
calibration, or label-conditioned selection.

The fixed seeds are 1111–1115. The anchor is selected as the seed with minimum
validation J, with the lower numeric seed breaking an exact tie. Test metrics
never participate in anchor or method selection. ADPEP-All is permanently the
main method; ADPEP-57 is an ablation only.

## Projection

For anchor prediction `a` and PE5 prediction `e`, ADPEP constructs the maximal
connected float32 interval containing `a` whose evaluator decision signature is
constant. ADPEP-57 uses `(Acc7 class, Acc5 class)`. ADPEP-All uses
`(Acc7 class, Acc5 class, Acc2 class)`.

The output is the point in that interval closest to `e`. A closed boundary is
used exactly. An open boundary is represented by the adjacent interior float32
value from `np.nextafter`. The projection function accepts only anchor
prediction, PE5 prediction, dataset identifier, and variant; it has no label
argument.

Every output is re-evaluated. On a mismatch, one `nextafter` step toward the
anchor is attempted, followed by a recorded fallback to the anchor. An
unverified output cannot enter metric evaluation.

## Binding and immutability

Inputs are bound by split, mode, sample_index, and sample_id and sorted by
sample_index. Duplicates, missing samples, split mixing, mode mixing, and
NaN/Inf are rejected. All five member sample/label sets are checked after
prediction freezing.

The projection command reads prediction columns without labels, writes four
label-free prediction files, records their SHA-256 values, and freezes them.
Only the aggregation command subsequently loads labels and computes metrics.
All required Stage 9A input SHA-256 values are recorded before projection and
verified again after aggregation.

## Metrics and gates

Valid/test are reported for LAV, LA, LV, L, and MissingMacro. MissingMacro is the
arithmetic mean of LA/LV/L for each metric. J is
`0.5 * MAE_LAV + 0.5 * MissingMacro_MAE`.

ADPEP-All must reproduce Anchor Acc7, Acc5, Acc2, and F1 within `1e-12` for
every reported mode and split, backed by exact per-sample decision equality.
ADPEP-57 must exactly reproduce Anchor Acc7 and Acc5.

Regression retention for a lower-is-better quantity is
`(Anchor - ADPEP) / (Anchor - PE5)` when the PE5 gain is positive. Corr
retention reverses the direction. Undefined ratios are reported rather than
silently divided.

The final classification is one of FULL SUCCESS, PARTIAL SUCCESS,
CLASSIFICATION-ONLY SUCCESS, REGRESSION TRADE-OFF UNSUPPORTED, or ENGINEERING
FAILURE. Exact classification inheritance alone is not a performance claim.

## Formal commands

```bash
python eval_anchor_decision_preserving_ensemble.py \
  --dataset mosi \
  --seeds 1111 1112 1113 1114 1115 \
  --splits valid test \
  --modes LAV LA LV L \
  --variants adpep57 adpep_all \
  --input-root result/missing_baseline/cfcompat_prediction_ensemble_v1/mosi \
  --output-root result/missing_baseline/anchor_decision_preserving_ensemble_v1/mosi

python aggregate_anchor_decision_preserving_ensemble.py \
  --dataset mosi \
  --anchor-selection valid_j \
  --main-method adpep_all \
  --verify-decisions \
  --compute-retention
```
