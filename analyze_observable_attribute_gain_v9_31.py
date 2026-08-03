"""V9.31 post-hoc audit of V9.30 attribute-to-expert-gain alignment.

No model, router, threshold, or attribute is fitted. The script reads the saved
strict V9.30 outer predictions and asks whether each deployment-time attribute
ranks the true gain of its matching expert.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

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
    quantile_gain_rows,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether V9.30 observable attributes align with the true "
            "post-hoc gain of their matching experts."
        )
    )
    parser.add_argument(
        "--v930-dir",
        default=(
            "result/full_same_stack_nested_crossfit_v919/mosi/seed_1111/"
            "v930_observable_attribute_experts"
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--quantile-bins", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1131)
    parser.add_argument("--min-spearman", type=float, default=0.05)
    parser.add_argument("--min-win-auc", type=float, default=0.55)
    parser.add_argument("--min-top-gain", type=float, default=0.005)
    parser.add_argument("--min-top-bottom-lift", type=float, default=0.010)
    parser.add_argument("--required-positive-folds", type=int, default=4)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    local = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in local.columns:
        if pd.api.types.is_float_dtype(local[column]):
            local[column] = local[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return "\n".join(
        [
            "|" + "|".join(local.columns) + "|",
            "|" + "|".join(["---"] * len(local.columns)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in local.itertuples(index=False, name=None)
            ],
        ]
    )


def required_prediction_columns() -> set[str]:
    columns = {
        "outer_fold",
        "sample_id",
        "group_id",
        "label",
        "action_prediction_anchor",
        "prediction_old_v921_convex_shrinkage",
    }
    for expert in EXPERT_NAMES:
        columns.add(f"action_prediction_{expert}")
        columns.add(f"applicability_{expert}")
    return columns


def main():
    cli = parse_args()
    config = AlignmentConfigV931(
        top_fraction=cli.top_fraction,
        quantile_bins=cli.quantile_bins,
        bootstrap_repetitions=cli.bootstrap_repetitions,
        bootstrap_seed=cli.bootstrap_seed,
        min_spearman=cli.min_spearman,
        min_win_auc=cli.min_win_auc,
        min_top_gain=cli.min_top_gain,
        min_top_bottom_lift=cli.min_top_bottom_lift,
        required_positive_folds=cli.required_positive_folds,
    )
    config.validate()

    v930_dir = Path(cli.v930_dir)
    source_predictions = v930_dir / "v930_outer_predictions.csv"
    source_summary = v930_dir / "v930_summary.json"
    for path in (source_predictions, source_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else v930_dir / "v931_attribute_gain_alignment"
    )
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(source_predictions)
    missing = sorted(required_prediction_columns() - set(frame.columns))
    if missing:
        raise ValueError(f"V9.30 predictions missing columns: {missing}")
    if frame.empty or frame["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("V9.30 outer predictions are empty or duplicated")
    numeric_columns = sorted(
        required_prediction_columns() - {"sample_id", "group_id"}
    )
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("V9.30 predictions contain non-finite values")

    v930_summary = json.loads(source_summary.read_text(encoding="utf-8"))
    provenance = v930_summary.get("provenance", {})
    if provenance.get("attributes_use_true_labels") is not False:
        raise RuntimeError("V9.30 summary does not certify label-free attributes")
    if provenance.get("winner_router_present") is not False:
        raise RuntimeError("V9.30 summary unexpectedly contains a winner router")

    labels = frame["label"].to_numpy(dtype=np.float64)
    baselines = {
        "anchor": frame["action_prediction_anchor"].to_numpy(dtype=np.float64),
        "old_v921_convex_shrinkage": frame[
            "prediction_old_v921_convex_shrinkage"
        ].to_numpy(dtype=np.float64),
    }
    aggregate_rows = []
    fold_rows = []
    quantile_rows = []
    bootstrap_rows = []
    sample = frame[
        ["outer_fold", "sample_id", "group_id", "label"]
    ].copy()

    folds = sorted(int(value) for value in frame["outer_fold"].unique())
    for expert_index, expert in enumerate(EXPERT_NAMES):
        score = frame[f"applicability_{expert}"].to_numpy(dtype=np.float64)
        prediction = frame[f"action_prediction_{expert}"].to_numpy(
            dtype=np.float64
        )
        sample[f"attribute_{expert}"] = score
        sample[f"prediction_{expert}"] = prediction

        for baseline_index, baseline_name in enumerate(BASELINE_NAMES):
            gain = posthoc_gain(baselines[baseline_name], prediction, labels)
            sample[f"gain_{expert}_vs_{baseline_name}"] = gain
            sample[f"win_{expert}_vs_{baseline_name}"] = gain > 0.0

            aggregate = alignment_metrics(
                score, gain, config.top_fraction
            )
            aggregate_rows.append(
                {
                    "expert": expert,
                    "baseline": baseline_name,
                    **aggregate,
                }
            )
            quantile_rows.extend(
                quantile_gain_rows(
                    score,
                    gain,
                    config.quantile_bins,
                    expert=expert,
                    baseline=baseline_name,
                    outer_fold="all",
                )
            )
            bootstrap = grouped_bootstrap_alignment(
                score,
                gain,
                frame["group_id"].tolist(),
                config,
                seed=(
                    config.bootstrap_seed
                    + 10007 * (expert_index + 1)
                    + 1009 * (baseline_index + 1)
                ),
            )
            bootstrap_rows.append(
                {
                    "expert": expert,
                    "baseline": baseline_name,
                    **bootstrap,
                }
            )

            for outer_fold in folds:
                mask = (
                    frame["outer_fold"].to_numpy(dtype=np.int64)
                    == int(outer_fold)
                )
                local = alignment_metrics(
                    score[mask], gain[mask], config.top_fraction
                )
                fold_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "expert": expert,
                        "baseline": baseline_name,
                        **local,
                    }
                )
                quantile_rows.extend(
                    quantile_gain_rows(
                        score[mask],
                        gain[mask],
                        config.quantile_bins,
                        expert=expert,
                        baseline=baseline_name,
                        outer_fold=int(outer_fold),
                    )
                )

    aggregate_frame = pd.DataFrame(aggregate_rows).sort_values(
        ["baseline", "expert"]
    )
    fold_frame = pd.DataFrame(fold_rows).sort_values(
        ["baseline", "expert", "outer_fold"]
    )
    quantile_frame = pd.DataFrame(quantile_rows).sort_values(
        ["baseline", "expert", "outer_fold", "quantile_bin"],
        key=lambda column: column.astype(str)
        if column.name == "outer_fold"
        else column,
    )
    bootstrap_frame = pd.DataFrame(bootstrap_rows).sort_values(
        ["baseline", "expert"]
    )

    decisions = {}
    anchor_status = {}
    strong_baseline_rows = {}
    for expert in EXPERT_NAMES:
        anchor_row = aggregate_frame[
            (aggregate_frame["expert"] == expert)
            & (aggregate_frame["baseline"] == "anchor")
        ].iloc[0].to_dict()
        anchor_bootstrap = bootstrap_frame[
            (bootstrap_frame["expert"] == expert)
            & (bootstrap_frame["baseline"] == "anchor")
        ].iloc[0].to_dict()
        positive_folds = int(
            (
                fold_frame[
                    (fold_frame["expert"] == expert)
                    & (fold_frame["baseline"] == "anchor")
                ]["top_gain"]
                > 0.0
            ).sum()
        )
        decision = classify_expert_alignment(
            anchor_row, anchor_bootstrap, positive_folds, config
        )
        decisions[expert] = {
            **decision,
            "positive_top_gain_folds": positive_folds,
        }
        anchor_status[expert] = str(decision["status"])
        strong_baseline_rows[expert] = aggregate_frame[
            (aggregate_frame["expert"] == expert)
            & (
                aggregate_frame["baseline"]
                == "old_v921_convex_shrinkage"
            )
        ].iloc[0].to_dict()

    verdict = make_overall_verdict(anchor_status, strong_baseline_rows)
    source_manifest = {
        "v930_predictions": {
            "path": str(source_predictions),
            "sha256": file_sha256(source_predictions),
        },
        "v930_summary": {
            "path": str(source_summary),
            "sha256": file_sha256(source_summary),
        },
    }
    summary = {
        "version": AUDIT_VERSION,
        "method": "posthoc_observable_attribute_to_matching_expert_gain_alignment",
        "v930_dir": str(v930_dir),
        "output_dir": str(output),
        "sample_count": int(len(frame)),
        "outer_fold_count": int(frame["outer_fold"].nunique()),
        "expert_names": list(EXPERT_NAMES),
        "baseline_names": list(BASELINE_NAMES),
        "config": asdict(config),
        "expert_decisions": decisions,
        "overall_verdict": verdict,
        "source_manifest": source_manifest,
        "provenance": {
            "new_models_trained": False,
            "winner_router_trained": False,
            "attributes_modified_or_refit": False,
            "labels_used_only_for_posthoc_gain_diagnostics": True,
            "outer_predictions_are_read_only": True,
            "decision_uses_anchor_alignment": True,
            "old_v921_is_reported_as_strong_baseline_check": True,
        },
    }

    aggregate_frame.to_csv(
        output / "v931_alignment_aggregate.csv", index=False
    )
    fold_frame.to_csv(output / "v931_alignment_by_fold.csv", index=False)
    quantile_frame.to_csv(output / "v931_quantile_gain.csv", index=False)
    bootstrap_frame.to_csv(
        output / "v931_group_bootstrap_alignment_ci.csv", index=False
    )
    sample.to_csv(output / "v931_sample_diagnostics.csv", index=False)
    (output / "v931_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    anchor_table = aggregate_frame[
        aggregate_frame["baseline"] == "anchor"
    ].copy()
    anchor_table["status"] = anchor_table["expert"].map(anchor_status)
    v921_table = aggregate_frame[
        aggregate_frame["baseline"] == "old_v921_convex_shrinkage"
    ].copy()
    decision_rows = [
        {
            "expert": expert,
            "status": decisions[expert]["status"],
            "criteria_passed": decisions[expert]["criteria_passed"],
            "positive_top_gain_folds": decisions[expert][
                "positive_top_gain_folds"
            ],
        }
        for expert in EXPERT_NAMES
    ]
    report = [
        "# V9.31 Observable Attribute-to-Gain Alignment Audit",
        "",
        "No model or router is trained. Labels are used only after prediction to compute true diagnostic gain.",
        "",
        "## Alignment versus V9.30 Anchor",
        "",
        markdown_table(
            anchor_table,
            [
                "expert",
                "status",
                "spearman",
                "win_auc",
                "gain_mean",
                "top_gain",
                "bottom_gain",
                "top_bottom_gain_lift",
                "top_win_rate",
            ],
        ),
        "",
        "## Alignment versus old V9.21",
        "",
        markdown_table(
            v921_table,
            [
                "expert",
                "spearman",
                "win_auc",
                "gain_mean",
                "top_gain",
                "bottom_gain",
                "top_bottom_gain_lift",
                "top_win_rate",
            ],
        ),
        "",
        "## Fold consistency and decision",
        "",
        markdown_table(
            pd.DataFrame(decision_rows),
            [
                "expert",
                "status",
                "criteria_passed",
                "positive_top_gain_folds",
            ],
        ),
        "",
        f"- Overall verdict: `{verdict['verdict']}`",
        f"- Negative-correlation training recommended: `{verdict['negative_correlation_recommended']}`",
        "",
        "Only attributes aligned with matching-expert gain should be retained. An unsupported result means redesign the attribute or expert before adding diversity losses.",
    ]
    (output / "v931_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.31 ATTRIBUTE-GAIN ALIGNMENT AUDIT COMPLETE")
    print("new models trained: False")
    print("winner router trained: False")
    for expert in EXPERT_NAMES:
        local = anchor_table[anchor_table["expert"] == expert].iloc[0]
        print(
            expert,
            "status=", anchor_status[expert],
            "spearman=", f"{float(local['spearman']):+.6f}",
            "auc=", f"{float(local['win_auc']):.6f}",
            "top_gain=", f"{float(local['top_gain']):+.6f}",
            "lift=", f"{float(local['top_bottom_gain_lift']):+.6f}",
        )
    print("verdict:", verdict["verdict"])
    print(
        "negative-correlation recommended:",
        verdict["negative_correlation_recommended"],
    )
    print("report:", output / "v931_report.md")


if __name__ == "__main__":
    main()
