#!/usr/bin/env python3
"""Synthetic CPU smoke test for Stage23D-A split, geometry, and probes."""

import json

import numpy as np
import pandas as pd

from stage23d_phase1 import (
    choose_classifier,
    choose_regression,
    geometry_features,
    predict_model,
    predict_probability,
    standardize_uncertainty,
)
from stage23d_self_risk_common import (
    OUT,
    atomic_json,
    risk_labels,
    source_stratified_split,
    utc_now,
)


def main():
    rng = np.random.RandomState(23170)
    sources = []
    for source_index in range(90):
        for clip in range(3):
            sources.append(
                {
                    "sample_id": f"source{source_index}_clip{clip}",
                    "video_id": f"source{source_index}",
                    "label": float(rng.normal()),
                }
            )
    frame = pd.DataFrame(sources)
    roles = source_stratified_split(frame, 23170)
    frame = frame.merge(
        roles[["video_id", "self_risk_role"]],
        on="video_id",
        validate="many_to_one",
    )
    raw = rng.normal(size=(len(frame), 48))
    latent_risk = np.abs(raw[:, 0]) + 0.3 * np.abs(raw[:, 1])
    prediction = frame["label"].to_numpy() + rng.normal(
        scale=0.15 + 0.25 * latent_risk
    )
    train_mask = frame["self_risk_role"].to_numpy() == "inner_train"
    valid_mask = frame["self_risk_role"].to_numpy() == "inner_valid"
    geometry, _, _ = geometry_features(
        raw,
        frame["video_id"].astype(str).to_numpy(),
        train_mask,
        16,
    )
    geometry["active_head_std"] = np.abs(raw[:, 0])
    geometry["submode_prediction_variance"] = np.square(raw[:, 1])
    uncertainty, _ = standardize_uncertainty(geometry, train_mask)
    target = pd.DataFrame(
        {
            "mode": "LAV",
            "self_risk_role": frame["self_risk_role"],
            "label": frame["label"],
            "prediction": prediction,
            "raw_uncertainty": uncertainty,
        }
    )
    target, thresholds = risk_labels(target)
    features = geometry.to_numpy(dtype=float)
    regression, candidates, _ = choose_regression(
        features,
        target["log_abs_error"].to_numpy(),
        train_mask,
        valid_mask,
        [("ridge", 1.0), ("hgb", "fixed")],
    )
    classifier, class_candidates, _ = choose_classifier(
        features,
        target["bad20"].to_numpy(),
        train_mask,
        valid_mask,
    )
    predicted_log = predict_model(regression, features[valid_mask])
    probability = predict_probability(classifier, features[valid_mask])
    finite = bool(
        np.isfinite(predicted_log).all()
        and np.isfinite(probability).all()
        and np.isfinite(features).all()
    )
    source_role_count = frame.groupby("video_id")["self_risk_role"].nunique()
    report = {
        "stage": "Stage23D-A synthetic CPU smoke",
        "status": "PASS" if finite and source_role_count.max() == 1 else "FAIL",
        "samples": len(frame),
        "sources": frame["video_id"].nunique(),
        "feature_dimensions": features.shape[1],
        "source_leakage": int((source_role_count > 1).sum()),
        "nan_inf_count": int(features.size - np.isfinite(features).sum()),
        "regression_candidates": candidates.to_dict(orient="records"),
        "classification_candidates": class_candidates.to_dict(orient="records"),
        "risk_thresholds": thresholds.to_dict(orient="records"),
        "completed_at": utc_now(),
    }
    path = OUT / "smoke" / "cpu_smoke_report.json"
    atomic_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise RuntimeError("Stage23D-A CPU smoke failed")


if __name__ == "__main__":
    main()
