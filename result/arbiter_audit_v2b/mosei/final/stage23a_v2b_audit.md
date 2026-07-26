# Stage23A-v2b Source-Aware Local Competence Audit

## Conclusion

**LOCAL_COMPETENCE_WEAK**

Stage23A-v2a remains `SIGNAL_AUDIT_FAIL`; this audit does not revise it.
The formal Feature-Rich Judge, Official Valid, Test, Experts and Student all
remained locked.

## Frozen choices

```json
{
  "A": {
    "common_configuration": {
      "DRS": "Local-DW",
      "K": 30,
      "distance": "standardized_euclidean",
      "inner_valid_J": 0.5821097800670738,
      "safe_coverage": 0.2,
      "space": "C",
      "tau": 0.05,
      "weighting": "reciprocal_epsilon_1e-6"
    },
    "development_samples": {
      "inner_train": 29752,
      "inner_valid": 4376
    },
    "family_best_methods": {
      "Local-DS": {
        "DRS": "Local-DS",
        "inner_valid_J": 0.5849171787944611,
        "margin": null,
        "tau": null
      },
      "Local-DW": {
        "DRS": "Local-DW",
        "inner_valid_J": 0.5785449347312149,
        "margin": null,
        "tau": 0.05
      },
      "Local-DWS": {
        "DRS": "Local-DWS",
        "inner_valid_J": 0.5785696627250438,
        "margin": 0.1,
        "tau": 0.05
      }
    },
    "neighbor_ledger_path": "/code/DLF-mosei-arbiter-audit-v1/result/arbiter_audit_v2b/mosei/development/direction_A_selected_inner_valid_neighbors.csv.gz",
    "neighbor_ledger_sha256": "a67de0d429f12e06519952364d88641d5cfd1a131126ead605cda311b607241d",
    "retrieval_stats_path": "/code/DLF-mosei-arbiter-audit-v1/result/arbiter_audit_v2b/mosei/protocol/direction_A_retrieval_stats.json",
    "retrieval_stats_sha256": "3cad4d5cf232eab5c004b3eca455325e8c54c3be0e0ee3da5faa3a2449b36d0c",
    "risk_ledger_path": "/code/DLF-mosei-arbiter-audit-v1/result/arbiter_audit_v2b/mosei/data/direction_A_expert_risk_ledger.csv.gz",
    "risk_ledger_sha256": "d490e3b199462ffa431ea26a59862eebe1310f4858a98eb3d367130a4ed47e30",
    "safety": {
      "coverage": 0.1,
      "estimated_gain_threshold": 0.01485855595938701,
      "inner_valid_J": 0.5782248456475392,
      "strong_static_inner_valid_J": 0.5812330817750744
    },
    "selected_method": {
      "DRS": "Local-DW",
      "inner_valid_J": 0.5785449347312149,
      "margin": null,
      "method_config_id": "Local-DW__tau0.05__marginNone",
      "tau": 0.05
    },
    "selected_region": {
      "K": 30,
      "beta": 0.25,
      "config_id": "H__cosine__K30__reciprocal_epsilon_1e-6__beta0p25",
      "distance": "cosine",
      "space": "H",
      "weighting": "reciprocal_epsilon_1e-6"
    },
    "strong_static": "per_mode_constrained_fixed_stacking"
  },
  "B": {
    "common_configuration": {
      "DRS": "Local-DW",
      "K": 30,
      "distance": "standardized_euclidean",
      "inner_valid_J": 0.5259201470241763,
      "safe_coverage": 0.2,
      "space": "C",
      "tau": 0.05,
      "weighting": "reciprocal_epsilon_1e-6"
    },
    "development_samples": {
      "inner_train": 27732,
      "inner_valid": 3444
    },
    "family_best_methods": {
      "Local-DS": {
        "DRS": "Local-DS",
        "inner_valid_J": 0.5311587327774226,
        "margin": null,
        "tau": null
      },
      "Local-DW": {
        "DRS": "Local-DW",
        "inner_valid_J": 0.526067182897588,
        "margin": null,
        "tau": 0.05
      },
      "Local-DWS": {
        "DRS": "Local-DWS",
        "inner_valid_J": 0.5262386658319056,
        "margin": 0.1,
        "tau": 0.02
      }
    },
    "neighbor_ledger_path": "/code/DLF-mosei-arbiter-audit-v1/result/arbiter_audit_v2b/mosei/development/direction_B_selected_inner_valid_neighbors.csv.gz",
    "neighbor_ledger_sha256": "000e7a8576a49f7b204239134335991c8d5ab11f339f2d14dd88dfb70c188698",
    "retrieval_stats_path": "/code/DLF-mosei-arbiter-audit-v1/result/arbiter_audit_v2b/mosei/protocol/direction_B_retrieval_stats.json",
    "retrieval_stats_sha256": "6151e341ccedf73af8d196390984335a93f115195fab4f7fd8f958dd661176ee",
    "risk_ledger_path": "/code/DLF-mosei-arbiter-audit-v1/result/arbiter_audit_v2b/mosei/data/direction_B_expert_risk_ledger.csv.gz",
    "risk_ledger_sha256": "38ffb937ae43f9d91ed208f043f2742c9bb12de4ea3cf94166921fb82d40c8f4",
    "safety": {
      "coverage": 0.5,
      "estimated_gain_threshold": -0.005052099403020094,
      "inner_valid_J": 0.5258240562747956,
      "strong_static_inner_valid_J": 0.5267324869574934
    },
    "selected_method": {
      "DRS": "Local-DW",
      "inner_valid_J": 0.526067182897588,
      "margin": null,
      "method_config_id": "Local-DW__tau0.05__marginNone",
      "tau": 0.05
    },
    "selected_region": {
      "K": 60,
      "beta": 0.5,
      "config_id": "H__standardized_euclidean__K60__reciprocal_epsilon_1e-6__beta0p5",
      "distance": "standardized_euclidean",
      "space": "H",
      "weighting": "reciprocal_epsilon_1e-6"
    },
    "strong_static": "per_mode_constrained_fixed_stacking"
  }
}
```

## Outer one-shot core metrics

| direction | method | J | MAE | Corr | Acc7 | Acc5 | Acc2 | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | strong_static | 0.553889 | 0.554211 | 0.740121 | 0.524987 | 0.540833 | 0.845751 | 0.845435 |
| A | selected_local_dynamic | 0.553827 | 0.554483 | 0.739123 | 0.528451 | 0.544425 | 0.848879 | 0.848442 |
| A | selected_safe_local | 0.553803 | 0.554298 | 0.738952 | 0.526815 | 0.542789 | 0.846198 | 0.845817 |
| B | strong_static | 0.555049 | 0.555281 | 0.758277 | 0.522123 | 0.545447 | 0.845220 | 0.844747 |
| B | selected_local_dynamic | 0.555179 | 0.555633 | 0.757248 | 0.521800 | 0.545066 | 0.847331 | 0.846546 |
| B | selected_safe_local | 0.554874 | 0.555325 | 0.758059 | 0.521654 | 0.544978 | 0.845069 | 0.844530 |

## Local regularity

| direction | method | spearman_local_risk_vs_error | spearman_local_regret_vs_regret | top1_accuracy | top2_coverage | soft_regret_KL |
| --- | --- | --- | --- | --- | --- | --- |
| A | selected_true_neighborhood | 0.196243 | 0.300726 | 0.202560 | 0.379683 | 1.012147 |
| A | N0_random_same_mode_source__seed23401 | -0.007255 | -0.000003 | 0.199801 | 0.401206 | 0.956977 |
| A | N0_random_same_mode_source__seed23402 | -0.001689 | -0.001200 | 0.201597 | 0.400019 | 0.956910 |
| A | N0_random_same_mode_source__seed23403 | -0.014638 | 0.004093 | 0.202046 | 0.403900 | 0.954754 |
| A | N1_shuffled_content__seed23411 | 0.162685 | 0.292462 | 0.199480 | 0.376283 | 1.029614 |
| A | N1_shuffled_content__seed23412 | 0.160340 | 0.290547 | 0.189376 | 0.368136 | 1.025047 |
| A | N1_shuffled_content__seed23413 | 0.159900 | 0.294880 | 0.199160 | 0.377983 | 1.029883 |
| A | N4_decision_randomization__seed23421 | 0.088371 | 0.027067 | 0.196273 | 0.388793 | 1.014613 |
| A | N4_decision_randomization__seed23422 | 0.049394 | 0.022842 | 0.193578 | 0.391936 | 1.011755 |
| A | N4_decision_randomization__seed23423 | 0.026044 | 0.031553 | 0.183827 | 0.387381 | 0.998229 |
| A | N2_global_competence | 0.005701 | -0.003060 | 0.195503 | 0.385168 | 0.844991 |
| A | N3_mode_only_competence | 0.005850 | -0.003137 | 0.195503 | 0.385168 | 0.845020 |
| B | selected_true_neighborhood | 0.247474 | 0.288424 | 0.215366 | 0.406616 | 0.877613 |
| B | N0_random_same_mode_source__seed23401 | 0.004887 | 0.005109 | 0.203176 | 0.400229 | 0.875768 |
| B | N0_random_same_mode_source__seed23402 | -0.006222 | -0.004179 | 0.199865 | 0.398851 | 0.879562 |
| B | N0_random_same_mode_source__seed23403 | -0.008758 | -0.002119 | 0.198195 | 0.396537 | 0.878108 |
| B | N1_shuffled_content__seed23411 | 0.202610 | 0.284089 | 0.212201 | 0.407290 | 0.882660 |
| B | N1_shuffled_content__seed23412 | 0.212066 | 0.284292 | 0.214575 | 0.404594 | 0.883047 |
| B | N1_shuffled_content__seed23413 | 0.191516 | 0.285200 | 0.217329 | 0.412770 | 0.879643 |
| B | N4_decision_randomization__seed23421 | 0.054685 | 0.070639 | 0.212641 | 0.419568 | 0.877034 |
| B | N4_decision_randomization__seed23422 | 0.061619 | 0.049996 | 0.213256 | 0.414205 | 0.890632 |
| B | N4_decision_randomization__seed23423 | 0.128393 | 0.036063 | 0.188350 | 0.382794 | 0.897138 |
| B | N2_global_competence | 0.003270 | 0.010332 | 0.211967 | 0.405034 | 0.809788 |
| B | N3_mode_only_competence | 0.004415 | 0.009980 | 0.200891 | 0.405034 | 0.809822 |

## Promotion gate

```json
{
  "mean_delta_J_vs_strong_static": -0.00013065014999996682,
  "mean_delta_J_gate_pass": false,
  "direction_delta_J": {
    "A": -8.627809999994795e-05,
    "B": -0.0001750221999999857
  },
  "worst_direction_delta_J": -8.627809999994795e-05,
  "worst_direction_gate_pass": true,
  "mean_delta_J_vs_strongest_random_or_shuffled": -0.00026931149999998016,
  "control_gate_pass": false,
  "missing_modes_improved": 1,
  "missing_mode_gate_pass": false,
  "corr_delta_by_direction": [
    -0.0011698548999999892,
    -0.00021746709999992397
  ],
  "corr_gate_pass": true,
  "classification_mean_deltas": {
    "Acc7": 0.0006797531000000134,
    "Acc5": 0.0007439050499999933,
    "Acc2": 0.00014803990000006317,
    "F1": 8.236984999993036e-05
  },
  "classification_gate_pass": true,
  "high_confidence_trigger_gate_pass": false,
  "direction_consistency_gate_pass": true,
  "not_global_or_mode_only_gate_pass": true,
  "outer_one_shot_gate_pass": true
}
```

## Source-aware neighborhood summary

```json
[
  {
    "direction": "A",
    "role": "outer_evaluation",
    "queries": 31176,
    "neighbor_rows": 935280,
    "unique_neighbor_sources": 1006,
    "top1_source_share": 0.00788533915,
    "top10_source_share": 0.05909139509,
    "mean_nearest_distance": 0.1756502378,
    "mean_kth_distance": 0.3061886379
  },
  {
    "direction": "B",
    "role": "outer_evaluation",
    "queries": 34128,
    "neighbor_rows": 2047680,
    "unique_neighbor_sources": 980,
    "top1_source_share": 0.006696847164,
    "top10_source_share": 0.05260685263,
    "mean_nearest_distance": 0.4900987227,
    "mean_kth_distance": 0.694125236
  }
]
```

## Plain-language questions

1. Local regularity is judged by held-out risk/regret Spearman and ranking, not
   by Oracle space.
2. The winning region is recorded independently for Directions A/B above.
3. True neighborhoods are compared against all random/shuffled controls; the
   gate uses the strongest negative control.
4. DS/DW/DWS were selected only by inner-valid J.
5. Safe deferral is audited at 10/20/30/50/100% coverage and by one frozen
   inner-valid threshold.
6. Direction agreement is an explicit gate.
7. Final promotion status is `LOCAL_COMPETENCE_WEAK`.
8. Density and source dominance distinguish lack of regularity from sparse or
   monopolized neighborhoods.
9. If FAIL, the frozen protocol permanently closes post-hoc Judge work on this
   Expert pool and redirects work to structurally specialized joint training.
