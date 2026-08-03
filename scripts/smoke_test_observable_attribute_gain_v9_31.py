"""Synthetic smoke tests for V9.31 attribute-to-gain audit utilities."""

from __future__ import annotations

import numpy as np

from trains.singleTask.observable_attribute_gain_audit_v931 import (
    AlignmentConfigV931,
    alignment_metrics,
    binary_auc,
    classify_expert_alignment,
    grouped_bootstrap_alignment,
    make_overall_verdict,
    posthoc_gain,
    quantile_gain_rows,
)


def main():
    rng = np.random.default_rng(931)
    n = 240
    groups = np.asarray([f"g{index // 6:03d}" for index in range(n)])
    labels = rng.normal(size=n)
    baseline = labels + rng.normal(scale=0.55, size=n)

    score = np.linspace(0.0, 1.0, n)
    expert = baseline.copy()
    expert -= np.sign(baseline - labels) * (0.24 * score - 0.08)
    gain = posthoc_gain(baseline, expert, labels)

    config = AlignmentConfigV931(
        bootstrap_repetitions=200,
        required_positive_folds=4,
    )
    config.validate()
    metrics = alignment_metrics(score, gain, config.top_fraction)
    assert metrics["spearman"] > 0.85
    assert metrics["win_auc"] > 0.90
    assert metrics["top_gain"] > 0.10
    assert metrics["top_bottom_gain_lift"] > 0.15
    assert binary_auc(score, gain > 0.0) > 0.90

    bootstrap = grouped_bootstrap_alignment(
        score, gain, groups, config, seed=931
    )
    assert bootstrap["top_gain_ci_low"] > 0.0
    decision = classify_expert_alignment(
        metrics, bootstrap, positive_top_folds=5, config=config
    )
    assert decision["status"] == "supported"

    bins = quantile_gain_rows(
        score,
        gain,
        bins=5,
        expert="audio_informative",
        baseline="anchor",
        outer_fold="all",
    )
    assert len(bins) == 5
    assert bins[-1]["mean_gain"] > bins[0]["mean_gain"]

    inverse_metrics = alignment_metrics(1.0 - score, gain, config.top_fraction)
    inverse_bootstrap = grouped_bootstrap_alignment(
        1.0 - score, gain, groups, config, seed=932
    )
    inverse = classify_expert_alignment(
        inverse_metrics,
        inverse_bootstrap,
        positive_top_folds=0,
        config=config,
    )
    assert inverse["status"] == "unsupported"

    verdict = make_overall_verdict(
        {
            "text_stable": "supported",
            "audio_informative": "supported",
            "vision_informative": "unsupported",
            "cross_modal_conflict": "unsupported",
        },
        {
            "text_stable": {"top_gain": 0.02, "top_bottom_gain_lift": 0.03},
            "audio_informative": {
                "top_gain": 0.03,
                "top_bottom_gain_lift": 0.04,
            },
            "vision_informative": {
                "top_gain": -0.01,
                "top_bottom_gain_lift": 0.00,
            },
            "cross_modal_conflict": {
                "top_gain": -0.02,
                "top_bottom_gain_lift": -0.01,
            },
        },
    )
    assert (
        verdict["verdict"]
        == "alignment_supports_targeted_diversity_training"
    )
    assert verdict["negative_correlation_recommended"] is True

    print("V9.31 SMOKE TEST PASSED")
    print("spearman:", f"{metrics['spearman']:.6f}")
    print("win_auc:", f"{metrics['win_auc']:.6f}")
    print("top_gain:", f"{metrics['top_gain']:+.6f}")
    print("top_bottom_lift:", f"{metrics['top_bottom_gain_lift']:+.6f}")


if __name__ == "__main__":
    main()
