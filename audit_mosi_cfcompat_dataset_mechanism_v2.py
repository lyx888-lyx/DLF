"""Hardened independent-audit entry point."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_mosi_cfcompat_dataset_mechanism as base
from trains.singleTask.mosi_cfcompat_audit_utils import (
    FORMAL_SEEDS,
    MODES,
    sha256_file,
)
from trains.singleTask.mosi_cfcompat_audit_v2_utils import (
    joint_video_bootstrap,
    mechanism_assessment,
    opportunity_ranking,
    prediction_events,
)


def main():
    cli = base.parse_args()
    base.prediction_events = prediction_events
    base.joint_video_bootstrap = joint_video_bootstrap
    base.opportunity_ranking = opportunity_ranking
    base.mechanism_assessment = mechanism_assessment
    base.main()

    root = Path(cli.result_dir)
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    check_path = root / "independent_audit_check.json"
    payload = json.loads(check_path.read_text(encoding="utf-8"))

    cache_binding = True
    for record in manifest["source_records"]:
        for path_key, sha_key in (
            ("compatibility_cache_csv", "compatibility_cache_csv_sha256"),
            ("compatibility_cache_config", "compatibility_cache_config_sha256"),
        ):
            path = Path(record[path_key])
            cache_binding = bool(
                cache_binding
                and path.is_file()
                and sha256_file(path) == record[sha_key]
            )

    dataset_source = manifest.get("dataset_source", {})
    dataset_binding = True
    for path_key, sha_key in (
        ("feature_path", "feature_sha256"),
        ("config_path", "config_sha256"),
    ):
        path = Path(str(dataset_source.get(path_key, "")))
        dataset_binding = bool(
            dataset_binding
            and path.is_file()
            and sha256_file(path) == dataset_source.get(sha_key)
        )

    samples = pd.read_csv(root / "dataset_samples_train_valid.csv")
    events = pd.read_csv(root / "valid_prediction_events.csv")
    train = samples.loc[samples.Split.astype(str).eq("train")].copy()
    valid = samples.loc[samples.Split.astype(str).eq("valid")].copy()
    official_counts = bool(
        len(train) == 1284
        and len(valid) == 229
        and not train.sample_index.duplicated().any()
        and not valid.sample_index.duplicated().any()
        and train.sample_id.astype(str).nunique() == len(train)
        and valid.sample_id.astype(str).nunique() == len(valid)
    )

    prediction_binding = True
    expected = valid[
        ["sample_index", "sample_id", "video_id", "label"]
    ].sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    for seed in FORMAL_SEEDS:
        for mode in MODES:
            local = events.loc[
                events.Seed.astype(int).eq(seed)
                & events.Mode.astype(str).eq(mode),
                ["sample_index", "sample_id", "video_id", "label"],
            ].sort_values("sample_index", kind="mergesort").reset_index(drop=True)
            prediction_binding = bool(
                prediction_binding
                and len(local) == len(expected)
                and np.array_equal(
                    local.sample_index.to_numpy(dtype=int),
                    expected.sample_index.to_numpy(dtype=int),
                )
                and local.sample_id.astype(str).equals(
                    expected.sample_id.astype(str)
                )
                and local.video_id.astype(str).equals(
                    expected.video_id.astype(str)
                )
                and np.allclose(
                    local.label.to_numpy(dtype=float),
                    expected.label.to_numpy(dtype=float),
                    atol=1e-12,
                    rtol=0.0,
                )
            )

    payload["checks"]["compatibility_cache_binding"] = cache_binding
    payload["checks"]["dataset_and_config_source_binding"] = dataset_binding
    payload["checks"]["official_mosi_train_valid_identity"] = official_counts
    payload["checks"]["valid_sample_prediction_identity_binding"] = prediction_binding
    payload["passed"] = bool(all(payload["checks"].values()))
    check_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not payload["passed"]:
        failed = [key for key, value in payload["checks"].items() if not value]
        raise RuntimeError("Hardened source binding failed: {}".format(failed))
    print("compatibility_cache_binding: True")
    print("dataset_and_config_source_binding: True")
    print("official_mosi_train_valid_identity: True")
    print("valid_sample_prediction_identity_binding: True")


if __name__ == "__main__":
    main()
