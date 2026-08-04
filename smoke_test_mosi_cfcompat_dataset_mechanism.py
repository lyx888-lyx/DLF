"""Synthetic smoke tests for the MOSI CFCompat dataset-mechanism audit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trains.singleTask.mosi_cfcompat_audit_utils import (
    FORMAL_SEEDS,
    MODES,
    dataset_limitation_flags,
    dataset_split_summary,
    empirical_compatibility,
    group_mechanism_summary,
    mechanism_assessment,
    modality_marginal_value,
    overall_prediction_summary,
    split_shift_summary,
)
from trains.singleTask.mosi_cfcompat_audit_v2_utils import (
    intensity_label,
    joint_video_bootstrap,
    opportunity_ranking,
    prediction_events,
    sentiment_bin,
)


def synthetic_samples():
    rows = []
    for split, videos, per_video in (("train", 8, 14), ("valid", 4, 10)):
        for video in range(videos):
            for segment in range(per_video):
                label = float(((segment + video) % 7) - 3)
                rows.append(
                    {
                        "Split": split,
                        "sample_index": len(rows),
                        "sample_id": f"{split}_v{video}|{segment}",
                        "video_id": f"{split}_v{video}",
                        "segment_id": str(segment),
                        "segment_order": float(segment),
                        "label": label,
                        "sentiment_bin": int(label),
                        "polarity": "negative" if label < -0.5 else ("positive" if label > 0.5 else "neutral"),
                        "intensity": "strong" if abs(label) >= 2.5 else ("medium" if abs(label) >= 1.5 else ("weak" if abs(label) >= 0.5 else "neutral")),
                        "normalized_text": f"synthetic sentence {segment % 5}",
                        "char_count": 20,
                        "token_count": 3,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_events(samples):
    valid = samples.loc[samples.Split.eq("valid")].reset_index(drop=True)
    rows = []
    for seed in FORMAL_SEEDS:
        for mode_index, mode in enumerate(MODES):
            for row in valid.itertuples(index=False):
                compatibility = 0.05 + 0.90 * ((row.sample_index % 10) + 0.5) / 10.0
                baseline = row.label * 0.70 + 0.20 + 0.02 * mode_index
                teacher = row.label * 0.92
                improvement = 0.12 * compatibility
                cfcompat = baseline + improvement * np.sign(teacher - baseline)
                rows.append(
                    {
                        "Seed": int(seed),
                        "Mode": mode,
                        "sample_index": int(row.sample_index),
                        "sample_id": row.sample_id,
                        "video_id": row.video_id,
                        "segment_id": row.segment_id,
                        "label": float(row.label),
                        "baseline_prediction": float(baseline),
                        "cfcompat_prediction": float(cfcompat),
                        "teacher_prediction": float(teacher),
                        "evaluator_LAV_prediction": float(row.label * 0.8),
                        "evaluator_shift": float(1.0 - compatibility),
                        "compatibility_proxy": float(compatibility),
                        "token_count": int(row.token_count),
                    }
                )
    return prediction_events(pd.DataFrame(rows))


def main():
    labels = np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=float)
    expected_bins = np.asarray([-3, -2, -1, 0, 1, 2, 3], dtype=int)
    expected_intensity = np.asarray(
        ["strong", "medium", "weak", "neutral", "weak", "medium", "strong"]
    )
    np.testing.assert_array_equal(sentiment_bin(labels), expected_bins)
    np.testing.assert_array_equal(sentiment_bin(pd.Series(labels)), expected_bins)
    np.testing.assert_array_equal(intensity_label(labels), expected_intensity)
    np.testing.assert_array_equal(
        intensity_label(pd.Series(labels)), expected_intensity
    )

    compatibility = empirical_compatibility([0.1, 0.2, 0.3, 0.4], [0.05, 0.25, 0.50])
    assert compatibility[0] > compatibility[1] > compatibility[2]
    assert np.all((compatibility > 0) & (compatibility < 1))

    samples = synthetic_samples()
    split = dataset_split_summary(samples)
    shift = split_shift_summary(samples)
    assert set(split.Split) == {"train", "valid"}
    assert shift["train_valid_video_overlap_count"] == 0

    events = synthetic_events(samples)
    expected = len(FORMAL_SEEDS) * len(MODES) * int((samples.Split == "valid").sum())
    assert len(events) == expected
    overall = overall_prediction_summary(events)
    groups = group_mechanism_summary(events)
    modality = modality_marginal_value(events)
    bootstrap_a = joint_video_bootstrap(events, replicates=100, seed=123)
    bootstrap_b = joint_video_bootstrap(events, replicates=100, seed=123)
    pd.testing.assert_frame_equal(bootstrap_a, bootstrap_b)
    assert np.isfinite(bootstrap_a.select_dtypes(include=[np.number]).to_numpy()).all()
    opportunities = opportunity_ranking(groups)
    assert list(opportunities.columns)
    limitations = dataset_limitation_flags(split, shift, modality, overall)
    mechanism = mechanism_assessment(events, overall, bootstrap_a)
    assert "verdict" in mechanism
    assert "small_valid_video_support" in limitations
    print("MOSI CFCompatKD dataset-mechanism utility smoke test passed")


if __name__ == "__main__":
    main()
