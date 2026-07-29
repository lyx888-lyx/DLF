# MOSI V9.1 Staged Protocol

Use this protocol instead of the earlier non-staged V9.1 runner.

## Scientific change

The standalone region-balanced mixture checkpoint is selected only from its own
Validation predictions:

```text
direct mixture MAE
+ 0.05 * worst-region MAE
+ 0.05 * fold MAE standard deviation
```

Only after that expert checkpoint is frozen does the code run one compact
Validation-only attribution search over:

```text
legacy_reference
legacy_beta_search
zero_shrinkage
trained_mixture
```

This avoids multiplying the small MOSI Validation set across every
`epoch x committee x beta x gamma` combination.

## Run

```bash
bash scripts/run_mosi_decoupled_region_mixture_v91_staged.sh \
  2>&1 | tee mosi_decoupled_region_mixture_v91_staged.log
```

## Audit

```bash
python3 scripts/audit_decoupled_region_mixture_v91.py \
  --run-dir result/decoupled_region_mixture_v91/mosi/seed_1111 \
  --expected-reference-mae 0.6994764 \
  --reference-tolerance 0.0002
```

## Interpretation

Training contributes only when the audit reports:

```text
selected_policy=trained_mixture ... gamma>0
training_contributed=True
```

Otherwise the output identifies the actual source of the selected gain:
committee reweighting, zero shrinkage, or the unchanged V7.1 reference.

Run Seed 1111 only. Do not tune the grids after reading Test.
