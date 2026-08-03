"""Independent engineering and leakage audit for V9.31 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.observable_attribute_gain_audit_v931 import (
    AUDIT_VERSION,
    BASELINE_NAMES,
    EXPERT_NAMES,
    AlignmentConfigV931,
    alignment_metrics,
    classify_expert_alignment,
    grouped_bootstrap_alignment,
    make_overall_verdict,
    posthoc_gain,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default=(
            "result/full_same_stack_nested_crossfit_v919/mosi/seed_1111/"
            "v930_observable_attribute_experts/"
            "v931_attribute_gain_alignment"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=2e-6)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(left, right, tolerance: float, message: str) -> None:
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return
    if not np.isfinite(left_value) or not np.isfinite(right_value):
        raise AssertionError(
            f"{message}: non-finite values {left_value} / {right_value}"
        )
    if abs(left_value - right_value) > float(tolerance):
        raise AssertionError(
            f"{message}: {left_value} != {right_value}"
        )


def main():
    cli = parse_args()
    result_dir = Path(cli.result_dir)
    required = {
        "summary": result_dir / "v931_summary.json",
        "aggregate": result_dir / "v931_alignment_aggregate.csv",
        "fold": result_dir / "v931_alignment_by_fold.csv",
        "quantile": result_dir / "v931_quantile_gain.csv",
        "bootstrap": result_dir / "v931_group_bootstrap_alignment_ci.csv",
        "sample": result_dir / "v931_sample_diagnostics.csv",
        "report": result_dir / "v931_report.md",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V9.31 outputs: {missing}")

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    if summary.get("version") != AUDIT_VERSION:
        raise AssertionError("unexpected V9.31 version")
    provenance = summary.get("provenance", {})
    expected_provenance = {
        "new_models_trained": False,
        "winner_router_trained": False,
        "attributes_modified_or_refit": False,
        "labels_used_only_for_posthoc_gain_diagnostics": True,
        "outer_predictions_are_read_only": True,
        "decision_uses_anchor_alignment": True,
        "old_v921_is_reported_as_strong_baseline_check": True,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) is not expected:
            raise AssertionError(
                f"invalid provenance {key}: {provenance.get(key)}"
            )

    source_manifest = summary["source_manifest"]
    source_predictions = Path(
        source_manifest["v930_predictions"]["path"]
    )
    source_summary = Path(source_manifest["v930_summary"]["path"])
    for role, path in (
        ("v930_predictions", source_predictions),
        ("v930_summary", source_summary),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        stored = source_manifest[role]["sha256"]
        actual = file_sha256(path)
        if actual != stored:
            raise AssertionError(f"source checksum changed: {role}")

    v930_summary = json.loads(source_summary.read_text(encoding="utf-8"))
    v930_provenance = v930_summary.get("provenance", {})
    if v930_provenance.get("attributes_use_true_labels") is not False:
        raise AssertionError("source attributes are not certified label-free")
    if v930_provenance.get("winner_router_present") is not False:
        raise AssertionError("source unexpectedly contains a winner router")

    predictions = pd.read_csv(source_predictions)
    aggregate = pd.read_csv(required["aggregate"])
    fold_frame = pd.read_csv(required["fold"])
    bootstrap_frame = pd.read_csv(required["bootstrap"])
    sample = pd.read_csv(required["sample"])
    if predictions.empty or predictions["sample_id"].astype(str).duplicated().any():
        raise AssertionError("source outer predictions are empty or duplicated")
    if len(sample) != len(predictions):
        raise AssertionError("sample diagnostics row count mismatch")
    if not np.array_equal(
        sample["sample_id"].astype(str).to_numpy(),
        predictions["sample_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("sample diagnostic order changed")

    config = AlignmentConfigV931(**summary["config"])
    config.validate()
    labels = predictions["label"].to_numpy(dtype=np.float64)
    baselines = {
        "anchor": predictions["action_prediction_anchor"].to_numpy(
            dtype=np.float64
        ),
        "old_v921_convex_shrinkage": predictions[
            "prediction_old_v921_convex_shrinkage"
        ].to_numpy(dtype=np.float64),
    }
    aggregate_index = aggregate.set_index(["expert", "baseline"])
    fold_index = fold_frame.set_index(["outer_fold", "expert", "baseline"])
    bootstrap_index = bootstrap_frame.set_index(["expert", "baseline"])
    folds = sorted(int(value) for value in predictions["outer_fold"].unique())

    decisions = {}
    anchor_status = {}
    strong_baseline_rows = {}
    for expert_index, expert in enumerate(EXPERT_NAMES):
        score = predictions[f"applicability_{expert}"].to_numpy(
            dtype=np.float64
        )
        expert_prediction = predictions[
            f"action_prediction_{expert}"
        ].to_numpy(dtype=np.float64)
        if not np.allclose(
            sample[f"attribute_{expert}"].to_numpy(dtype=np.float64),
            score,
            atol=cli.tolerance,
            rtol=0.0,
        ):
            raise AssertionError(f"sample attribute changed for {expert}")
        if not np.allclose(
            sample[f"prediction_{expert}"].to_numpy(dtype=np.float64),
            expert_prediction,
            atol=cli.tolerance,
            rtol=0.0,
        ):
            raise AssertionError(f"sample prediction changed for {expert}")

        for baseline_index, baseline in enumerate(BASELINE_NAMES):
            gain = posthoc_gain(
                baselines[baseline], expert_prediction, labels
            )
            stored_gain = sample[
                f"gain_{expert}_vs_{baseline}"
            ].to_numpy(dtype=np.float64)
            if not np.allclose(
                gain, stored_gain, atol=cli.tolerance, rtol=0.0
            ):
                raise AssertionError(
                    f"sample gain does not recompute: {expert}/{baseline}"
                )
            recomputed = alignment_metrics(
                score, gain, config.top_fraction
            )
            stored = aggregate_index.loc[(expert, baseline)]
            for key, value in recomputed.items():
                assert_close(
                    value,
                    stored[key],
                    cli.tolerance,
                    f"aggregate {expert}/{baseline}/{key}",
                )

            recomputed_bootstrap = grouped_bootstrap_alignment(
                score,
                gain,
                predictions["group_id"].tolist(),
                config,
                seed=(
                    config.bootstrap_seed
                    + 10007 * (expert_index + 1)
                    + 1009 * (baseline_index + 1)
                ),
            )
            stored_bootstrap = bootstrap_index.loc[(expert, baseline)]
            for key, value in recomputed_bootstrap.items():
                assert_close(
                    value,
                    stored_bootstrap[key],
                    cli.tolerance,
                    f"bootstrap {expert}/{baseline}/{key}",
                )

            for outer_fold in folds:
                mask = (
                    predictions["outer_fold"].to_numpy(dtype=np.int64)
                    == int(outer_fold)
                )
                local = alignment_metrics(
                    score[mask], gain[mask], config.top_fraction
                )
                stored_fold = fold_index.loc[
                    (int(outer_fold), expert, baseline)
                ]
                for key, value in local.items():
                    assert_close(
                        value,
                        stored_fold[key],
                        cli.tolerance,
                        (
                            f"fold {outer_fold}/{expert}/"
                            f"{baseline}/{key}"
                        ),
                    )

        anchor_row = aggregate_index.loc[(expert, "anchor")].to_dict()
        anchor_bootstrap = bootstrap_index.loc[
            (expert, "anchor")
        ].to_dict()
        positive_folds = int(
            sum(
                float(
                    fold_index.loc[
                        (outer_fold, expert, "anchor"), "top_gain"
                    ]
                )
                > 0.0
                for outer_fold in folds
            )
        )
        decision = classify_expert_alignment(
            anchor_row, anchor_bootstrap, positive_folds, config
        )
        decisions[expert] = {
            **decision,
            "positive_top_gain_folds": positive_folds,
        }
        anchor_status[expert] = str(decision["status"])
        strong_baseline_rows[expert] = aggregate_index.loc[
            (expert, "old_v921_convex_shrinkage")
        ].to_dict()

    verdict = make_overall_verdict(anchor_status, strong_baseline_rows)
    if decisions != summary["expert_decisions"]:
        raise AssertionError("expert decisions do not recompute")
    if verdict != summary["overall_verdict"]:
        raise AssertionError("overall verdict does not recompute")

    expected_rows = len(EXPERT_NAMES) * len(BASELINE_NAMES)
    if len(aggregate) != expected_rows or len(bootstrap_frame) != expected_rows:
        raise AssertionError("aggregate/bootstrap row count mismatch")
    expected_fold_rows = expected_rows * len(folds)
    if len(fold_frame) != expected_fold_rows:
        raise AssertionError("fold row count mismatch")
    if required["quantile"].stat().st_size == 0 or required["report"].stat().st_size == 0:
        raise AssertionError("quantile table or report is empty")

    audit_result = {
        "version": AUDIT_VERSION,
        "passed": True,
        "result_dir": str(result_dir),
        "sample_count": int(len(predictions)),
        "outer_fold_count": int(len(folds)),
        "checks": {
            "source_checksums_verified": True,
            "v930_label_free_attribute_provenance_verified": True,
            "no_new_model_or_router_verified": True,
            "sample_gains_recomputed": True,
            "aggregate_alignment_recomputed": True,
            "fold_alignment_recomputed": True,
            "group_bootstrap_recomputed": True,
            "expert_decisions_recomputed": True,
            "overall_verdict_recomputed": True,
        },
    }
    (result_dir / "v931_audit_check.json").write_text(
        json.dumps(audit_result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("V9.31 OUTPUT AUDIT PASSED")
    print("samples:", len(predictions))
    print("outer folds:", len(folds))
    print("verdict:", verdict["verdict"])
    print("audit check:", result_dir / "v931_audit_check.json")


if __name__ == "__main__":
    main()
