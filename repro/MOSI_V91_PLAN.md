# MOSI V9.1: Decoupled Region-Balanced Mixture

## Why V9.1 exists

V9 selected epoch 0. Its small `~0.02` blend made the initialized mixture
approximately a zero-shrinkage path, while trained epochs did not improve the
Validation robust objective. Therefore the V9 Test gain cannot be attributed to
learned region-balanced experts.

V9.1 separates the questions:

1. Does changing the committee/Student weight help?
2. Does shrinking the Student toward zero help?
3. Does a mixture trained with region-balanced supervision add information?

## Main change

The new mixture is trained **directly**:

```text
mixture_value -> region-balanced Huber(label)
```

It is no longer trained through:

```text
legacy + 0.02 * (mixture - legacy)
```

The V7.1 backbone and residual Student remain frozen. The obsolete V9 blend
parameter is frozen and excluded from policy selection.

## Training objective

The direct mixture uses:

- Train-only inverse-square-root five-region weights;
- Huber regression on `mixture_value`;
- soft negative/neutral/positive gate supervision;
- polarity-specific expert regression;
- monotonic Acc-7 and Acc-5 auxiliary heads;
- ordinal/mixture consistency;
- distance-weighted cross-zero penalty;
- a very small marginal gate-balance regularizer.

GroupDRO remains disabled for the first run.

## Validation-only attribution families

V9.1 explicitly searches these families:

```text
legacy_reference
legacy_beta_search
zero_shrinkage
trained_mixture
```

Definitions:

```text
legacy_beta_search:
    beta * committee + (1-beta) * legacy

zero_shrinkage:
    beta * committee + (1-beta) * ((1-gamma) * legacy)

trained_mixture:
    beta * committee
    + (1-beta) * ((1-gamma) * legacy + gamma * trained_mixture)
```

Epoch 0 is never allowed to enter `trained_mixture`. This prevents the symmetric
zero initialization from masquerading as learned improvement.

Ordinary-positive Validation MAE is reported but is **not** part of the selection
objective because that group is too small for stable hyperparameter selection.

## Leakage discipline

- Region weights: Train labels only.
- New heads: Train only.
- Epoch, committee, beta, gamma and family: Validation only.
- Original V7.1 Hybrid: always a legal fallback.
- Test: evaluated once after the complete policy is frozen.

## Run

```bash
bash scripts/run_mosi_decoupled_region_mixture_v91.sh \
  2>&1 | tee mosi_decoupled_region_mixture_v91.log
```

Audit:

```bash
python3 scripts/audit_decoupled_region_mixture_v91.py \
  --run-dir result/decoupled_region_mixture_v91/mosi/seed_1111 \
  --expected-reference-mae 0.6994764 \
  --reference-tolerance 0.0002
```

## Files to inspect

```text
v91_train_region_weights.csv
v91_training_history.csv
v91_valid_policy_search.csv
v91_final_valid_policy.csv
v91_test_comparison.csv
v91_test_region_diagnostics.csv
v91_test_predictions.csv
decoupled_region_mixture_v91_summary.json
```

## Interpretation

The result counts as evidence for learned imbalance correction only when:

```text
selected_epoch > 0
selected_policy.source == "trained_mixture"
selected_policy.gamma > 0
training_contributed == true
```

Other outcomes have clear meanings:

- `legacy_reference`: no new Valid-supported gain.
- `legacy_beta_search`: gain comes from committee reweighting.
- `zero_shrinkage`: gain comes from output shrinkage, not learned experts.
- `trained_mixture`: the trained region-balanced expert contributes.

Run Seed 1111 only. Do not launch five seeds until the audit passes and the
selected family is correctly attributed.
