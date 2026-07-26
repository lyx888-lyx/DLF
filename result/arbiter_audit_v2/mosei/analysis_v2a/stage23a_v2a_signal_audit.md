# Stage23A-v2a Feature Extraction and Signal Audit

## Decision

**SIGNAL_AUDIT_FAIL**

This is a Train-OOF signal audit only. It does **not** authorize or train the
formal dual-head Judge. Official Valid, Test, Student training and Expert
modification remained locked.

## Strong-static-relative Oracle geometry (outer one-shot)

| direction | strong_static_J_or_MAE | oracle_J_or_MAE | oracle_minus_strong_static | fraction_gain_gt_0p01 | strong_static_strictly_better_fraction | best_second_gap_mean |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.553889 | 0.397753 | -0.156136 | 0.941365 | 0.043687 | 0.088730 |
| B | 0.555049 | 0.404660 | -0.150389 | 0.940840 | 0.045036 | 0.079928 |

## Hierarchical consistency transfer

| direction | median_rho_error | mean_AUROC_best | mean_AUROC_top_bottom |
| --- | --- | --- | --- |
| A | 0.055175 | 0.501065 | 0.552296 |
| B | 0.071399 | 0.498799 | 0.553920 |

Outer sign-reversal fractions: {"A": 0.0, "B": 0.0}.

## Incremental regret-probe signal

| direction | increment | delta_regret_MAE | delta_regret_Spearman | delta_ranking_accuracy | delta_soft_preference_KL | delta_soft_expected_regret |
| --- | --- | --- | --- | --- | --- | --- |
| A | C2-C1 | 0.000026 | -0.002230 | 0.000994 | 0.001516 | -0.000059 |
| A | C3-C2 | 0.000201 | -0.002661 | 0.005677 | 0.001222 | -0.000021 |
| B | C2-C1 | -0.000069 | -0.001422 | 0.004073 | 0.000773 | -0.000072 |
| B | C3-C2 | 0.000507 | -0.007800 | 0.015471 | 0.002169 | 0.000014 |

For error-like metrics (MAE, KL, expected regret), negative deltas improve.
For Spearman and ranking accuracy, positive deltas improve.

## Fold fingerprint diagnostic

| probe | role | selected_C_inner_valid_only | balanced_accuracy | AUROC | log_loss |
| --- | --- | --- | --- | --- | --- |
| content_only | inner_valid | 1.000000 | 0.491442 | 0.472746 | 0.693177 |
| consistency_only | inner_valid | 10.000000 | 0.727548 | 0.807068 | 0.541880 |
| content_plus_consistency | inner_valid | 0.100000 | 0.734324 | 0.810174 | 0.540432 |

A Direction-local fold classifier is undefined because a development Direction
contains only one fold. Therefore this diagnostic pools both source-disjoint
inner-train sets and evaluates only on both source-disjoint inner-valid sets.
No pseudo-outer score is reported, because the opposite Direction outer samples
are the same samples used on the pooled training side.

## Frozen controls

| direction | control | metric | mean | std | strongest |
| --- | --- | --- | --- | --- | --- |
| A | cross_source_shuffle | regret_MAE | 0.124153 | 0.000070 | 0.124085 |
| A | cross_source_shuffle | regret_Spearman | 0.338032 | 0.000857 | 0.339110 |
| A | cross_source_shuffle | ranking_accuracy | 0.153900 | 0.000568 | 0.154702 |
| A | cross_source_shuffle | soft_preference_KL | 0.290135 | 0.000452 | 0.289662 |
| A | cross_source_shuffle | soft_expected_regret | 0.168732 | 0.000038 | 0.168681 |
| A | matched_noise | regret_MAE | 0.124043 | 0.000013 | 0.124029 |
| A | matched_noise | regret_Spearman | 0.339163 | 0.000157 | 0.339385 |
| A | matched_noise | ranking_accuracy | 0.155451 | 0.000600 | 0.156274 |
| A | matched_noise | soft_preference_KL | 0.289476 | 0.000047 | 0.289411 |
| A | matched_noise | soft_expected_regret | 0.168689 | 0.000017 | 0.168675 |
| B | cross_source_shuffle | regret_MAE | 0.119731 | 0.000051 | 0.119694 |
| B | cross_source_shuffle | regret_Spearman | 0.324351 | 0.000598 | 0.324989 |
| B | cross_source_shuffle | ranking_accuracy | 0.152963 | 0.004263 | 0.158990 |
| B | cross_source_shuffle | soft_preference_KL | 0.257926 | 0.000173 | 0.257792 |
| B | cross_source_shuffle | soft_expected_regret | 0.162118 | 0.000050 | 0.162057 |
| B | matched_noise | regret_MAE | 0.119734 | 0.000032 | 0.119704 |
| B | matched_noise | regret_Spearman | 0.324189 | 0.000617 | 0.324723 |
| B | matched_noise | ranking_accuracy | 0.149506 | 0.001030 | 0.150727 |
| B | matched_noise | soft_preference_KL | 0.257916 | 0.000213 | 0.257760 |
| B | matched_noise | soft_expected_regret | 0.162142 | 0.000011 | 0.162134 |

## Criterion ledger

```json
{
  "broad_oracle_geometry": true,
  "consistency_transfers_without_systematic_reversal": true,
  "content_increment_C3_over_C2_both_directions": false,
  "aligned_content_beats_strongest_controls_both_directions": false,
  "outer_regret_ranking_signal_above_random_both_directions": false,
  "control_detail": {
    "A": false,
    "B": false
  },
  "fold_identity_AUROC": 0.810174465514569,
  "no_dominant_fold_fingerprint_AUROC_lt_0p80": false
}
```

## Locks and leakage

- Frozen Experts retrained or modified: **No**
- Formal Judge trained: **No**
- Official Valid accesses: **0**
- Locked Test accesses: **0**
- Student trained: **No**
- Outer labels used for fitting/selection: **No**
- 482 ineffective Vision clips excluded from Vision PCA fit and forced to zero
  after transform: **Yes**

## Plain-language answers

1. The one-best-existing-Expert Oracle remains far better than the inner-valid
   selected strong static baseline; the opportunity is not an artifact of a weak
   per-mode stacking reference.
2. The hierarchical consistency signal is summarized above, including
   Direction-wise outer reversals; it is useful only if its association transfers.
3. Content utility is judged by C3−C2 and by aligned content against every frozen
   shuffle/noise control, not by training fit.
4. Fold identity is explicitly audited. A high inner-valid AUROC is treated as a
   checkpoint/domain fingerprint warning, not as positive routing evidence.
5. All preprocessing and probe choices were made on Direction inner-train and
   inner-valid only; outer rows were transformed and scored once.
6. The conclusion is `SIGNAL_AUDIT_FAIL` and still requires a separate explicit user
   authorization before any formal Judge training.
