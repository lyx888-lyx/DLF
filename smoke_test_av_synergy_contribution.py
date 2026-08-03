"""Synthetic smoke test for the AV synergy contribution audit."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_av_synergy_contribution import AuditConfig, build_artifacts, write_artifacts


def make_frame(seed: int, split: str) -> pd.DataFrame:
    sample_ids = ["videoA$_$0", "videoA$_$1", "videoB$_$0", "videoB$_$1", "videoC$_$0", "videoC$_$1"]
    labels = np.array([-1.0, 1.0, -0.5, 0.8, -1.5, 1.5], dtype=float)
    l_pred = labels + np.array([0.4, -0.4, 0.3, -0.3, 0.5, -0.5])
    audio = np.array([-0.15, 0.15, -0.10, 0.10, -0.20, 0.20])
    vision = np.array([-0.10, 0.10, -0.05, 0.05, -0.15, 0.15])
    synergy = np.array([-0.10, 0.10, -0.10, 0.10, -0.10, 0.10])
    jitter = (seed - 1113) * 0.002
    la_pred = l_pred + audio + jitter
    lv_pred = l_pred + vision - jitter
    lav_pred = l_pred + audio + vision + synergy
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "sample_index": np.arange(len(sample_ids)),
            "label": labels,
            "LAV_pred": lav_pred,
            "LA_pred": la_pred,
            "LV_pred": lv_pred,
            "L_pred": l_pred,
            "Seed": seed,
            "Method": "Online",
            "Split": split,
            "SelectedBy": "validation_J",
        }
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for seed in range(1111, 1116):
            seed_dir = (
                root
                / "missing_baseline"
                / "cfcompat_stability_v1"
                / "mosi"
                / f"seed{seed}"
            )
            seed_dir.mkdir(parents=True)
            for split in ("valid", "test"):
                make_frame(seed, split).to_csv(
                    seed_dir / f"online_{split}_predictions.csv", index=False
                )
        config = AuditConfig(
            result_root=str(root),
            bootstrap_repetitions=200,
            required_synergy_gain=0.01,
            required_marginal_gain=0.01,
            required_sign_agreement=0.70,
            required_high_magnitude_coverage=0.50,
        )
        artifacts = build_artifacts(config)
        summary = artifacts["summary"]
        assert summary["synergy_gate"]["passed"] is True
        assert summary["marginal_gate"]["passed"] is True
        assert summary["verdict"] == (
            "SUPPORTED_BUILD_CONTRIBUTION_FACTORIZATION_DISTILLATION"
        )
        ensemble = artifacts["ensemble_samples"]
        expected = np.array([-0.10, 0.10, -0.10, 0.10, -0.10, 0.10] * 2)
        assert np.allclose(ensemble["synergy_effect"], expected)
        output = root / "audit"
        write_artifacts(artifacts, output)
        assert (output / "avsc_summary.json").is_file()
        assert (output / "avsc_report.md").is_file()
    print("AV SYNERGY CONTRIBUTION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
