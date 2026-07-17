# Stage 9A: CFCompatKD Cross-Seed Prediction Ensemble

## Locked method

- The members are exactly seeds 1111, 1112, 1113, 1114, and 1115.
- Every member is the Stage8 Online validation-selected Student checkpoint.
- Every scalar output has weight 0.2. The same weights are used for LAV, LA,
  LV, and L on valid and test.
- Model parameters and hidden states are never averaged.
- There is no calibration, weight learning, seed deletion, test-best
  checkpoint, EMA, Soup, or later-stage checkpoint.

The objective is:

`J = 0.5 * MAE_LAV + 0.5 * mean(MAE_LA, MAE_LV, MAE_L)`.

## Dual-path replay

The offline path loads the five Stage8 prediction CSVs, sorts each by
`sample_index`, validates exact `sample_id` and label identity, and averages
predictions. The online path sequentially loads each Student checkpoint,
re-runs valid/test inference for all four modes, releases the model and CUDA
cache, then averages the aligned outputs.

The maximum offline/online prediction difference and every metric difference
must be at most `1e-6`. The resulting objectives must replay Stage8 within
`1e-4`: valid `0.669356`, test `0.705777`.

## Audits

- Leave-one-out ensembles are diagnostic only and cannot select members.
- Pairwise prediction/error correlation and signed/absolute disagreement are
  reported for all ten model pairs.
- Sample evidence reports prediction range, standard deviation, mean
  individual error, ensemble error, and their difference.
- Compatibility quartiles are analysis-only. They use the frozen Stage2.5
  evaluator predictions for the same split and the Stage3 transform:
  `delta = |LAV - missing|`, empirical rank, `compatibility = 1 - q`.
  This source is sample/label-bound and is not required by deployment.
- Sample-level paired bootstrap uses exactly 2000 resamples and fixed seeds.

## Deployment boundary

Formal inference needs only the five Student checkpoints, the requested
dataset split, and one GPU. Models are loaded one at a time. Teacher,
counterfactual evaluator, compatibility cache, and train data are not runtime
dependencies.

The result is explicitly a five-model inference method, never a single-model
result.
