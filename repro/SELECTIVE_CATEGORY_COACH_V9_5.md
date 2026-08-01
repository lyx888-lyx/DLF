# V9.5 Selective Category Coach

V9.5 keeps the strongest frozen expert pool established in V9.3:

- strong negative: V9.2 strong-negative residual expert;
- ordinary negative: CFCompatKD Anchor;
- boundary: V9 boundary expert;
- ordinary positive: V9 positive expert;
- strong positive: V9.2 strong-positive residual expert.

It does not retrain these experts. The goal is to test the original hypothesis:
classify emotional strength, then call the corresponding expert.

## Conservative category coach

The anchor prediction is already a competitive ordinal score. V9.5 therefore
uses it as the mandatory baseline and permits only a bounded correction learned
from the four aligned function-space logits. Semantic boundaries remain fixed at
`-1.5, -0.5, 0.5, 1.5`.

The calibrator is trained and evaluated by source-video grouped cross-fitting on
the V9.2 Train OOF cache. Its OOF metrics are always reported beside the raw
Anchor-threshold metrics.

## Selective routing

A specialist may be used only when all configured tests pass:

1. category source predicts its designated region;
2. optional calibrated/Anchor category agreement;
3. minimum region probability;
4. minimum distance from a category boundary;
5. minimum frozen-expert self-confidence;
6. expert prediction moves toward the semantic region center when enabled;
7. correction direction is consistent with the region.

Otherwise the system remains at Anchor. The specialist prediction is blended
with Anchor by Validation-selected `beta`.

## Validation safety gate

Validation searches a small interpretable routing grid. A non-Anchor policy is
eligible only when:

- MAE gain is at least `0.0015` by default;
- the 20th percentile of a source-video group bootstrap gain is non-negative;
- activation is non-zero and no greater than 35%.

If no policy passes, the deployable result is exactly Anchor. This prevents the
`0.0005` Validation improvements seen in V9.3/V9.4 from forcing a Test route.

## Run

```bash
cd /code/AAAI/DLF 2>/dev/null || cd /code/DLF

git fetch origin
git checkout innovation/selective-category-coach-v9-5
git pull origin innovation/selective-category-coach-v9-5

python3 scripts/smoke_test_selective_category_coach_v9_5.py

GPU=0 SEED=1111 \
  bash scripts/run_mosi_selective_category_coach_v9_5.sh \
  2>&1 | tee mosi_selective_category_coach_v95_seed1111.log
```

## Outputs

```text
result/selective_category_coach_v95/mosi/seed_1111/
```

Important files:

```text
v95_region_oof_predictions.csv
v95_route_calibration.csv
v95_test_summary.csv
selective_category_coach_v95_predictions.csv
selective_category_coach_v95_summary.json
```

True-region and sample-oracle policies remain diagnostic upper bounds only.
