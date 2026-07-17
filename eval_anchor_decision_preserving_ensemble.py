"""Generate frozen, label-free Stage 9B ADPEP predictions."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.anchor_decision_projection import (
    SUPPORTED_DATASETS,
    evaluator_decisions,
    project_array,
    select_anchor_seed,
)


SEEDS = (1111, 1112, 1113, 1114, 1115)
SPLITS = ("valid", "test")
MODES = ("LAV", "LA", "LV", "L")
VARIANTS = ("adpep57", "adpep_all")
INPUT_FILES = (
    "ensemble_predictions_valid.csv",
    "ensemble_predictions_test.csv",
    "individual_model_metrics.csv",
    "individual_predictions_manifest.json",
    "checkpoint_manifest.json",
    "ensemble_metrics.csv",
    "pe5_full_metric_comparison.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 9B label-free ADPEP projection.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    if tuple(args.seeds) != SEEDS:
        parser.error("Stage 9B requires seeds 1111 1112 1113 1114 1115 in order.")
    if tuple(args.splits) != SPLITS:
        parser.error("Formal Stage 9B requires valid then test.")
    if tuple(args.modes) != MODES:
        parser.error("Formal Stage 9B requires LAV LA LV L in order.")
    if tuple(args.variants) != VARIANTS:
        parser.error("Formal Stage 9B requires adpep57 and adpep_all.")
    if args.input_root is None:
        args.input_root = str(
            Path("result")
            / "missing_baseline"
            / "cfcompat_prediction_ensemble_v1"
            / args.dataset
        )
    if args.output_root is None:
        args.output_root = str(
            Path("result")
            / "missing_baseline"
            / "anchor_decision_preserving_ensemble_v1"
            / args.dataset
        )
    return args


def _prediction_frame(path, split, method, use_labels=False):
    columns = [
        "sample_index",
        "sample_id",
        "LAV_pred",
        "LA_pred",
        "LV_pred",
        "L_pred",
        "Split",
        "Method",
    ]
    if use_labels:
        columns.insert(2, "label")
    frame = pd.read_csv(path, usecols=columns)
    if frame.sample_index.duplicated().any():
        raise ValueError("Duplicate sample_index in {}.".format(path))
    if set(frame.Split.astype(str)) != {split}:
        raise ValueError("Split mixing in {}.".format(path))
    if set(frame.Method.astype(str)) != {method}:
        raise ValueError("Method mismatch in {}.".format(path))
    numeric_columns = ["sample_index"] + [
        "{}_pred".format(mode) for mode in MODES
    ]
    if use_labels:
        numeric_columns.append("label")
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("NaN/Inf in {}.".format(path))
    frame["sample_index"] = frame.sample_index.astype(int)
    return frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def _bind_without_labels(anchor, pe5):
    if not np.array_equal(
        anchor.sample_index.to_numpy(), pe5.sample_index.to_numpy()
    ):
        raise RuntimeError("Anchor/PE5 sample_index binding differs.")
    if not np.array_equal(
        anchor.sample_id.astype(str).to_numpy(),
        pe5.sample_id.astype(str).to_numpy(),
    ):
        raise RuntimeError("Anchor/PE5 sample_id binding differs.")


def _input_manifest(input_root):
    records = []
    for name in INPUT_FILES:
        path = input_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {"Name": name, "Path": str(path), "SHA256": sha256(path), "Bytes": path.stat().st_size}
        )
    return records


def _anchor_selection(input_root, seeds):
    metrics = pd.read_csv(input_root / "individual_model_metrics.csv")
    rows = metrics.loc[
        metrics.Split.eq("valid")
        & metrics.Method.eq("Online")
        & metrics.Mode.eq("LAV"),
        ["Seed", "J"],
    ].to_dict("records")
    anchor_seed = select_anchor_seed(rows, seeds)
    by_seed = {int(row["Seed"]): float(row["J"]) for row in rows}
    test = metrics.loc[
        metrics.Split.eq("test")
        & metrics.Method.eq("Online")
        & metrics.Mode.eq("LAV")
        & metrics.Seed.eq(anchor_seed),
        "J",
    ]
    if len(test) != 1:
        raise RuntimeError("Anchor test J is missing or duplicated.")
    return {
        "Rule": "argmin_fixed_seeds_validation_J_then_lower_seed",
        "FixedSeeds": list(seeds),
        "AnchorSeed": anchor_seed,
        "ValidationJBySeed": {str(seed): by_seed[seed] for seed in seeds},
        "AnchorJValid": by_seed[anchor_seed],
        "AnchorJTestAtValidSelection": float(test.iloc[0]),
        "SelectedWithoutTest": True,
        "HardCodedAnchorSeed": False,
    }


def _member_prediction_path(input_root, prediction_manifest, seed, split):
    members = [
        row
        for row in prediction_manifest["Members"]
        if int(row["Seed"]) == int(seed)
    ]
    if len(members) != 1:
        raise RuntimeError("Anchor prediction manifest member is not unique.")
    record = members[0]["Predictions"][split]
    path = input_root / Path(record["Path"]).name
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != record["SHA256"]:
        raise RuntimeError("Anchor member prediction SHA differs.")
    return path


def _empty_fallback_frame():
    return pd.DataFrame(
        columns=[
            "Split",
            "Mode",
            "Variant",
            "sample_index",
            "sample_id",
            "AnchorPrediction",
            "PE5Prediction",
            "FinalPrediction",
            "Reason",
        ]
    )


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # The projection API is mechanically prevented from accepting labels.
    for function in (project_array,):
        if "label" in inspect.signature(function).parameters:
            raise RuntimeError("Projection API accepts forbidden label input.")

    input_records = _input_manifest(input_root)
    input_snapshot = {row["Name"]: row["SHA256"] for row in input_records}
    anchor_selection = _anchor_selection(input_root, args.seeds)
    anchor_seed = anchor_selection["AnchorSeed"]

    prediction_manifest = json.loads(
        (input_root / "individual_predictions_manifest.json").read_text()
    )
    if [int(row["Seed"]) for row in prediction_manifest["Members"]] != list(
        args.seeds
    ):
        raise RuntimeError("Individual prediction seed order differs.")
    if any(
        str(row.get("SelectedBy")) != "validation_J"
        for row in prediction_manifest["Members"]
    ):
        raise RuntimeError("An individual prediction was not validation-selected.")
    checkpoint_manifest = json.loads(
        (input_root / "checkpoint_manifest.json").read_text()
    )
    if [int(seed) for seed in checkpoint_manifest["Seeds"]] != list(args.seeds):
        raise RuntimeError("Checkpoint manifest seed order differs.")
    if (
        not checkpoint_manifest.get("OnlyValidationSelectedOnlineCheckpoints")
        or any(
            str(row.get("SelectedBy")) != "validation_J"
            for row in checkpoint_manifest["Members"]
        )
    ):
        raise RuntimeError("Checkpoint manifest permits non-validation selection.")

    diagnostics = []
    verification = []
    fallbacks = []
    output_records = []

    for split in args.splits:
        pe5 = _prediction_frame(
            input_root / "ensemble_predictions_{}.csv".format(split),
            split,
            "CFCompatKD-PE5",
            use_labels=False,
        )
        anchor = _prediction_frame(
            _member_prediction_path(
                input_root, prediction_manifest, anchor_seed, split
            ),
            split,
            "Online",
            use_labels=False,
        )
        _bind_without_labels(anchor, pe5)
        for variant in args.variants:
            output = anchor[["sample_index", "sample_id"]].copy()
            output["Split"] = split
            output["Method"] = "ADPEP-57" if variant == "adpep57" else "ADPEP-All"
            output["AnchorSeed"] = anchor_seed
            for mode in args.modes:
                anchor_values = anchor["{}_pred".format(mode)].to_numpy(
                    dtype=np.float32
                )
                pe5_values = pe5["{}_pred".format(mode)].to_numpy(dtype=np.float32)
                projected, details = project_array(
                    anchor_values, pe5_values, args.dataset, variant
                )
                output["{}_pred".format(mode)] = projected
                anchor_decisions = evaluator_decisions(
                    anchor_values, args.dataset
                )
                final_decisions = evaluator_decisions(projected, args.dataset)
                required = 2 if variant == "adpep57" else 3
                mismatch_counts = [
                    int(np.count_nonzero(left != right))
                    for left, right in zip(
                        anchor_decisions[:required], final_decisions[:required]
                    )
                ]
                if any(mismatch_counts):
                    raise RuntimeError(
                        "{} {} {} decision preservation failure.".format(
                            split, mode, variant
                        )
                    )
                pe5_decisions = evaluator_decisions(pe5_values, args.dataset)
                constraint_flags = np.column_stack(
                    [
                        pe5_decisions[index] != anchor_decisions[index]
                        for index in range(3)
                    ]
                )
                changed_from_pe5 = projected != pe5_values
                diagnostics.append(
                    {
                        "Split": split,
                        "Mode": mode,
                        "Variant": variant,
                        "SampleCount": len(projected),
                        "ProjectedSampleCount": int(changed_from_pe5.sum()),
                        "ProjectedSampleRatio": float(changed_from_pe5.mean()),
                        "PE5AlreadyFeasibleCount": int(
                            sum(detail.pe5_already_feasible for detail in details)
                        ),
                        "PE5AlreadyFeasibleRatio": float(
                            np.mean(
                                [
                                    detail.pe5_already_feasible
                                    for detail in details
                                ]
                            )
                        ),
                        "BoundaryAdjustedCount": int(
                            sum(detail.boundary_adjusted for detail in details)
                        ),
                        "BoundaryAdjustedRatio": float(
                            np.mean([detail.boundary_adjusted for detail in details])
                        ),
                        "FallbackToAnchorCount": int(
                            sum(detail.fallback_to_anchor for detail in details)
                        ),
                        "FallbackToAnchorRatio": float(
                            np.mean([detail.fallback_to_anchor for detail in details])
                        ),
                        "Acc7LimitedCount": int(constraint_flags[:, 0].sum()),
                        "Acc7LimitedRatio": float(constraint_flags[:, 0].mean()),
                        "Acc5LimitedCount": int(constraint_flags[:, 1].sum()),
                        "Acc5LimitedRatio": float(constraint_flags[:, 1].mean()),
                        "Acc2LimitedCount": int(constraint_flags[:, 2].sum()),
                        "Acc2LimitedRatio": float(constraint_flags[:, 2].mean()),
                        "MultipleBoundaryLimitedCount": int(
                            (constraint_flags.sum(axis=1) > 1).sum()
                        ),
                        "MultipleBoundaryLimitedRatio": float(
                            (constraint_flags.sum(axis=1) > 1).mean()
                        ),
                        "MeanAnchorToPE5Distance": float(
                            np.mean(np.abs(anchor_values - pe5_values))
                        ),
                        "MeanAnchorToADPEPDistance": float(
                            np.mean(np.abs(anchor_values - projected))
                        ),
                        "MeanADPEPToPE5Distance": float(
                            np.mean(np.abs(projected - pe5_values))
                        ),
                        "OutputEqualsAnchorCount": int(
                            (projected == anchor_values).sum()
                        ),
                        "OutputEqualsAnchorRatio": float(
                            (projected == anchor_values).mean()
                        ),
                        "OutputEqualsPE5Count": int((projected == pe5_values).sum()),
                        "OutputEqualsPE5Ratio": float(
                            (projected == pe5_values).mean()
                        ),
                    }
                )
                verification.append(
                    {
                        "Split": split,
                        "Mode": mode,
                        "Variant": variant,
                        "SampleCount": len(projected),
                        "Acc7MismatchCount": mismatch_counts[0],
                        "Acc5MismatchCount": mismatch_counts[1],
                        "Acc2MismatchCount": mismatch_counts[2]
                        if variant == "adpep_all"
                        else 0,
                        "MaxVerificationDiscrepancy": max(mismatch_counts),
                        "Passed": not any(mismatch_counts),
                    }
                )
                for index, detail in enumerate(details):
                    if detail.fallback_to_anchor:
                        fallbacks.append(
                            {
                                "Split": split,
                                "Mode": mode,
                                "Variant": variant,
                                "sample_index": int(output.sample_index.iloc[index]),
                                "sample_id": str(output.sample_id.iloc[index]),
                                "AnchorPrediction": float(anchor_values[index]),
                                "PE5Prediction": float(pe5_values[index]),
                                "FinalPrediction": float(projected[index]),
                                "Reason": detail.fallback_reason,
                            }
                        )
            ordered = [
                "sample_index",
                "sample_id",
                "LAV_pred",
                "LA_pred",
                "LV_pred",
                "L_pred",
                "Split",
                "Method",
                "AnchorSeed",
            ]
            output = output[ordered].sort_values(
                "sample_index", kind="mergesort"
            )
            filename = "{}_predictions_{}.csv".format(variant, split)
            path = output_root / filename
            output.to_csv(path, index=False, float_format="%.9g")
            output_records.append(
                {
                    "Variant": variant,
                    "Split": split,
                    "Path": str(path),
                    "SHA256": sha256(path),
                    "Rows": len(output),
                    "LabelsPresent": False,
                    "FrozenBeforeMetricLabelsLoaded": True,
                }
            )

    pd.DataFrame(diagnostics).to_csv(
        output_root / "adpep_boundary_diagnostics.csv", index=False
    )
    pd.DataFrame(verification).to_csv(
        output_root / "adpep_decision_verification.csv", index=False
    )
    (
        pd.DataFrame(fallbacks)
        if fallbacks
        else _empty_fallback_frame()
    ).to_csv(output_root / "projection_fallback_samples.csv", index=False)

    (output_root / "anchor_selection.json").write_text(
        json.dumps(anchor_selection, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "input_manifest_with_sha.json").write_text(
        json.dumps(
            {
                "Dataset": args.dataset,
                "Files": input_records,
                "ProjectionDidNotLoadLabels": True,
                "InputHashesRecordedBeforeProjection": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_manifest_path = output_root / "output_prediction_manifest.json"
    output_manifest_path.write_text(
        json.dumps(
            {
                "Dataset": args.dataset,
                "AnchorSeed": anchor_seed,
                "MainMethod": "adpep_all",
                "AblationOnly": "adpep57",
                "Predictions": output_records,
                "LabelsPresent": False,
                "PredictionsFrozenBeforeMetricLabelsLoaded": True,
                "ProjectionAPIAcceptsLabel": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    current = {row["Name"]: sha256(input_root / row["Name"]) for row in input_records}
    if current != input_snapshot:
        raise RuntimeError("Stage 9A input SHA changed during projection.")
    print(
        "ADPEP label-free projection passed: anchor_seed={} outputs={} fallbacks={}".format(
            anchor_seed, len(output_records), len(fallbacks)
        )
    )


if __name__ == "__main__":
    main()
