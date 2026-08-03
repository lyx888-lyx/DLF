"""Synthetic smoke tests for the V9.29 expert-pool viability audit utilities."""

from __future__ import annotations

import numpy as np

from trains.singleTask.expert_pool_viability_audit_v929 import (
    compute_fold_upper_bounds,
    fit_mae_simplex,
    make_viability_verdict,
    mae,
    oracle_gap_closure,
)


def main():
    rng = np.random.default_rng(929)
    action_names = ("anchor", "strong_negative", "boundary", "positive", "strong_positive")

    labels = np.concatenate(
        [rng.normal(-1.0, 0.35, size=80), rng.normal(1.0, 0.35, size=80)]
    )
    anchor = labels + rng.normal(0.0, 0.55, size=len(labels))
    negative = labels + np.where(labels < 0.0, rng.normal(0.0, 0.18, len(labels)), 0.55)
    boundary = labels + rng.normal(0.0, 0.42, size=len(labels))
    positive = labels + np.where(labels >= 0.0, rng.normal(0.0, 0.18, len(labels)), -0.55)
    strong_positive = labels + rng.normal(0.12, 0.48, size=len(labels))
    actions = np.column_stack([anchor, negative, boundary, positive, strong_positive])

    bounds = compute_fold_upper_bounds(actions, labels, action_names)
    best_single_mae = mae(bounds["best_single_prediction"], labels)
    convex_mae = mae(bounds["convex_prediction"], labels)
    oracle_mae = mae(bounds["sample_oracle_prediction"], labels)
    assert convex_mae <= best_single_mae + 1e-7
    assert oracle_mae <= best_single_mae + 1e-7
    assert len(bounds["sample_oracle_action"]) == len(labels)
    assert np.isfinite(bounds["convex_weights"]).all()
    assert abs(float(bounds["convex_weights"].sum()) - 1.0) < 1e-9

    pooled_weights = fit_mae_simplex(actions, labels)
    assert pooled_weights.shape == (len(action_names),)
    assert np.all(pooled_weights >= -1e-12)
    assert abs(float(pooled_weights.sum()) - 1.0) < 1e-9

    closure = oracle_gap_closure(
        mae(anchor, labels), convex_mae, oracle_mae
    )
    assert np.isfinite(closure)

    verdict = make_viability_verdict(
        {
            "v921_convex_shrinkage": {"mae": 0.722},
            "best_global_fixed_action_cheating": {"mae": 0.718},
            "fold_best_single_cheating": {"mae": 0.710},
            "pooled_global_convex_cheating": {"mae": 0.705},
            "fold_local_convex_cheating": {"mae": 0.699},
            "sample_oracle": {"mae": 0.620},
        },
        target_mae=0.70,
    )
    assert (
        verdict["verdict"]
        == "global_fixed_fusion_cannot_hit_target_fold_specific_cheating_can"
    )
    assert verdict["below_target"]["sample_oracle"] is True

    print("V9.29 SMOKE TEST PASSED")
    print("best_single_action:", bounds["best_single_action"])
    print("best_single_mae:", f"{best_single_mae:.6f}")
    print("convex_mae:", f"{convex_mae:.6f}")
    print("sample_oracle_mae:", f"{oracle_mae:.6f}")
    print("convex_weights:", bounds["convex_weights"].tolist())


if __name__ == "__main__":
    main()
