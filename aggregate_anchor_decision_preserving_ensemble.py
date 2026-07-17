"""Aggregate frozen Stage 9B ADPEP predictions after the label-free gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_anchor_decision_preserving_ensemble import (
    MODES,
    SEEDS,
    SPLITS,
    _member_prediction_path,
    sha256,
)
from trains.singleTask.anchor_decision_projection import (
    SUPPORTED_DATASETS,
    evaluator_decisions,
    retention_ratio,
)
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    METRICS,
    metrics_from_predictions,
)


METHODS = ("Anchor", "PE5", "ADPEP-57", "ADPEP-All", "Online5Mean")
CLASSIFICATION_METRICS = ("acc_7", "acc_5", "acc_2", "F1_score")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate frozen Stage 9B ADPEP.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="mosi")
    parser.add_argument("--anchor-selection", choices=("valid_j",), default="valid_j")
    parser.add_argument("--main-method", choices=("adpep_all",), default="adpep_all")
    parser.add_argument("--verify-decisions", action="store_true")
    parser.add_argument("--compute-retention", action="store_true")
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    if not args.verify_decisions or not args.compute_retention:
        parser.error("Formal Stage 9B requires decision and retention verification.")
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


def _verify_input_hashes(input_root, output_root):
    manifest = json.loads((output_root / "input_manifest_with_sha.json").read_text())
    for record in manifest["Files"]:
        path = input_root / record["Name"]
        if sha256(path) != record["SHA256"]:
            raise RuntimeError("Input SHA changed: {}.".format(record["Name"]))
    return manifest


def _verify_frozen_outputs(output_root):
    manifest = json.loads((output_root / "output_prediction_manifest.json").read_text())
    if (
        not manifest.get("PredictionsFrozenBeforeMetricLabelsLoaded")
        or not manifest.get("ProjectionAPIAcceptsLabel") is False
    ):
        raise RuntimeError("Prediction freezing or label-free gate is absent.")
    for record in manifest["Predictions"]:
        path = output_root / Path(record["Path"]).name
        if sha256(path) != record["SHA256"]:
            raise RuntimeError("Frozen output SHA changed: {}.".format(path))
        columns = pd.read_csv(path, nrows=0).columns
        if "label" in columns:
            raise RuntimeError("Frozen projected prediction contains labels.")
    return manifest


def _load_source_frame(path, split, method):
    frame = pd.read_csv(path)
    required = {
        "sample_index",
        "sample_id",
        "label",
        "Split",
        "Method",
        *["{}_pred".format(mode) for mode in MODES],
    }
    if not required.issubset(frame.columns):
        raise ValueError("Source prediction columns are incomplete.")
    if frame.sample_index.duplicated().any():
        raise ValueError("Duplicate sample_index in source predictions.")
    if set(frame.Split.astype(str)) != {split} or set(frame.Method.astype(str)) != {
        method
    }:
        raise ValueError("Source split/method binding differs.")
    numeric = frame[
        ["sample_index", "label"]
        + ["{}_pred".format(mode) for mode in MODES]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Source predictions contain NaN/Inf.")
    frame["sample_index"] = frame.sample_index.astype(int)
    return frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def _bind_identity(reference, candidate, include_label=True):
    if not np.array_equal(
        reference.sample_index.to_numpy(), candidate.sample_index.to_numpy()
    ):
        raise RuntimeError("sample_index binding differs.")
    if not np.array_equal(
        reference.sample_id.astype(str).to_numpy(),
        candidate.sample_id.astype(str).to_numpy(),
    ):
        raise RuntimeError("sample_id binding differs.")
    if include_label and not np.array_equal(
        reference.label.to_numpy(dtype=np.float32),
        candidate.label.to_numpy(dtype=np.float32),
    ):
        raise RuntimeError("label binding differs.")


def _load_projected(output_root, variant, split, labels):
    path = output_root / "{}_predictions_{}.csv".format(variant, split)
    frame = pd.read_csv(path)
    if "label" in frame:
        raise RuntimeError("Projected file unexpectedly contains labels.")
    if frame.sample_index.duplicated().any():
        raise ValueError("Duplicate projected sample_index.")
    if set(frame.Split.astype(str)) != {split}:
        raise ValueError("Projected split mixing.")
    frame["sample_index"] = frame.sample_index.astype(int)
    frame = frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    _bind_identity(labels, frame, include_label=False)
    frame.insert(2, "label", labels.label.to_numpy(dtype=np.float32))
    numeric = frame[
        ["sample_index", "label"]
        + ["{}_pred".format(mode) for mode in MODES]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Projected predictions contain NaN/Inf.")
    return frame


def _metric_rows(method, split, frame):
    metrics, objective = metrics_from_predictions(frame)
    return [
        {
            "Split": split,
            "Method": method,
            "Mode": mode,
            "J": objective,
            **metrics[mode],
        }
        for mode in MODES + ("MissingMacro",)
    ]


def _online_mean_rows(individual, split):
    rows = []
    selected = individual.loc[
        individual.Split.eq(split)
        & individual.Method.eq("Online")
        & individual.Seed.isin(SEEDS)
    ]
    for mode in MODES + ("MissingMacro",):
        local = selected.loc[selected.Mode.eq(mode)]
        if set(local.Seed.astype(int)) != set(SEEDS):
            raise RuntimeError("Online5Mean seed set is incomplete.")
        row = {
            "Split": split,
            "Method": "Online5Mean",
            "Mode": mode,
            "J": float(local.J.mean()),
        }
        row.update({metric: float(local[metric].mean()) for metric in METRICS})
        rows.append(row)
    return rows


def _verify_decisions(dataset, split, source, projected, variant):
    rows = []
    required = 2 if variant == "adpep57" else 3
    for mode in MODES:
        anchor = evaluator_decisions(
            source["{}_pred".format(mode)].to_numpy(dtype=np.float32), dataset
        )
        final = evaluator_decisions(
            projected["{}_pred".format(mode)].to_numpy(dtype=np.float32), dataset
        )
        mismatch = [
            int(np.count_nonzero(left != right))
            for left, right in zip(anchor[:required], final[:required])
        ]
        if any(mismatch):
            raise RuntimeError("DECISION PRESERVATION FAILURE")
        rows.append((split, mode, variant, mismatch))
    return rows


def _full_comparison(metrics):
    indexed = metrics.set_index(["Split", "Mode", "Method"])
    rows = []
    for split in SPLITS:
        for mode in MODES + ("MissingMacro",):
            for metric in tuple(METRICS) + ("J",):
                values = {
                    method: float(indexed.loc[(split, mode, method), metric])
                    for method in METHODS
                }
                rows.append(
                    {
                        "Split": split,
                        "Mode": mode,
                        "Metric": metric,
                        **values,
                        "DeltaADPEPAllVsAnchor": values["ADPEP-All"]
                        - values["Anchor"],
                        "DeltaADPEPAllVsPE5": values["ADPEP-All"] - values["PE5"],
                        "DeltaADPEP57VsAnchor": values["ADPEP-57"]
                        - values["Anchor"],
                    }
                )
    return pd.DataFrame(rows)


def _retention_rows(metrics):
    indexed = metrics.set_index(["Split", "Mode", "Method"])
    rows = []
    quantities = (
        ("J", "LAV", "J", True),
        ("LAV_MAE", "LAV", "MAE", True),
        ("MissingMacro_MAE", "MissingMacro", "MAE", True),
        ("LAV_Corr", "LAV", "Corr", False),
        ("MissingMacro_Corr", "MissingMacro", "Corr", False),
    )
    for split in SPLITS:
        for variant in ("ADPEP-57", "ADPEP-All"):
            for name, mode, metric, lower in quantities:
                anchor = float(indexed.loc[(split, mode, "Anchor"), metric])
                pe5 = float(indexed.loc[(split, mode, "PE5"), metric])
                method = float(indexed.loc[(split, mode, variant), metric])
                pe5_gain, method_gain, ratio = retention_ratio(
                    anchor, pe5, method, lower_is_better=lower
                )
                rows.append(
                    {
                        "Split": split,
                        "Variant": variant,
                        "Quantity": name,
                        "Direction": "LowerIsBetter" if lower else "HigherIsBetter",
                        "Anchor": anchor,
                        "PE5": pe5,
                        "ADPEP": method,
                        "GainPE5": pe5_gain,
                        "GainADPEP": method_gain,
                        "RetentionRatio": 0.0 if ratio is None else ratio,
                        "Defined": ratio is not None,
                        "UndefinedReason": "Applicable"
                        if ratio is not None
                        else "PE5 gain is not positive",
                    }
                )
    return pd.DataFrame(rows)


def _classification_gate(metrics):
    indexed = metrics.set_index(["Split", "Mode", "Method"])
    rows = []
    for split in SPLITS:
        for mode in MODES + ("MissingMacro",):
            for metric in CLASSIFICATION_METRICS:
                anchor = float(indexed.loc[(split, mode, "Anchor"), metric])
                all_value = float(indexed.loc[(split, mode, "ADPEP-All"), metric])
                difference = abs(all_value - anchor)
                passed = difference <= 1e-12
                rows.append(
                    {
                        "Split": split,
                        "Mode": mode,
                        "Metric": metric,
                        "Anchor": anchor,
                        "ADPEPAll": all_value,
                        "AbsoluteDifference": difference,
                        "Tolerance": 1e-12,
                        "Passed": passed,
                    }
                )
                if not passed:
                    raise RuntimeError("DECISION PRESERVATION FAILURE")
    return pd.DataFrame(rows)


def _classification(metrics, retention):
    indexed = metrics.set_index(["Split", "Mode", "Method"])
    test = lambda mode, method, metric: float(
        indexed.loc[("test", mode, method), metric]
    )
    decision_pass = True
    basic = (
        test("LAV", "ADPEP-All", "J") < test("LAV", "Anchor", "J")
        and test("LAV", "ADPEP-All", "MAE") < test("LAV", "Anchor", "MAE")
        and test("MissingMacro", "ADPEP-All", "MAE")
        < test("MissingMacro", "Anchor", "MAE")
        and test("LAV", "ADPEP-All", "Corr") >= test("LAV", "Anchor", "Corr")
        and test("MissingMacro", "ADPEP-All", "Corr")
        >= test("MissingMacro", "Anchor", "Corr")
        and decision_pass
    )
    required = retention.loc[
        retention.Variant.eq("ADPEP-All")
        & retention.Split.eq("test")
        & retention.Quantity.isin(("J", "LAV_MAE", "MissingMacro_MAE"))
    ]
    ratios_pass = bool(
        len(required) == 3
        and required.Defined.all()
        and (required.RetentionRatio >= 0.50).all()
    )
    direction_consistent = all(
        float(indexed.loc[(split, mode, "ADPEP-All"), metric])
        < float(indexed.loc[(split, mode, "Anchor"), metric])
        for split in SPLITS
        for mode, metric in (
            ("LAV", "J"),
            ("LAV", "MAE"),
            ("MissingMacro", "MAE"),
        )
    )
    if basic and ratios_pass and direction_consistent:
        return "FULL SUCCESS", basic, ratios_pass, direction_consistent
    if basic:
        return "PARTIAL SUCCESS", basic, ratios_pass, direction_consistent
    tied = all(
        abs(
            float(indexed.loc[("test", mode, "ADPEP-All"), metric])
            - float(indexed.loc[("test", mode, "Anchor"), metric])
        )
        <= 1e-12
        for mode, metric in (
            ("LAV", "J"),
            ("LAV", "MAE"),
            ("MissingMacro", "MAE"),
        )
    )
    if tied:
        return "CLASSIFICATION-ONLY SUCCESS", basic, ratios_pass, direction_consistent
    return (
        "REGRESSION TRADE-OFF UNSUPPORTED",
        basic,
        ratios_pass,
        direction_consistent,
    )


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    _verify_input_hashes(input_root, output_root)
    frozen_manifest = _verify_frozen_outputs(output_root)
    anchor_seed = int(frozen_manifest["AnchorSeed"])
    prediction_manifest = json.loads(
        (input_root / "individual_predictions_manifest.json").read_text()
    )

    frames = {}
    verification_rows = []
    all_metric_rows = []
    individual = pd.read_csv(input_root / "individual_model_metrics.csv")
    for split in SPLITS:
        pe5 = _load_source_frame(
            input_root / "ensemble_predictions_{}.csv".format(split),
            split,
            "CFCompatKD-PE5",
        )
        anchor = _load_source_frame(
            _member_prediction_path(
                input_root, prediction_manifest, anchor_seed, split
            ),
            split,
            "Online",
        )
        _bind_identity(pe5, anchor)
        # Verify every fixed member has the same split/sample/label set.
        for seed in SEEDS:
            member = _load_source_frame(
                _member_prediction_path(
                    input_root, prediction_manifest, seed, split
                ),
                split,
                "Online",
            )
            _bind_identity(pe5, member)
        adpep57 = _load_projected(output_root, "adpep57", split, pe5)
        adpep_all = _load_projected(output_root, "adpep_all", split, pe5)
        verification_rows.extend(
            _verify_decisions(args.dataset, split, anchor, adpep57, "adpep57")
        )
        verification_rows.extend(
            _verify_decisions(args.dataset, split, anchor, adpep_all, "adpep_all")
        )
        frames[split] = {
            "Anchor": anchor,
            "PE5": pe5,
            "ADPEP-57": adpep57,
            "ADPEP-All": adpep_all,
        }
        for method, frame in frames[split].items():
            all_metric_rows.extend(_metric_rows(method, split, frame))
        all_metric_rows.extend(_online_mean_rows(individual, split))

    metrics = pd.DataFrame(all_metric_rows)
    if not np.isfinite(metrics.select_dtypes(include=[np.number])).all().all():
        raise FloatingPointError("ADPEP metrics contain NaN/Inf.")

    # Replay existing Anchor/PE5 metrics rather than trusting saved aggregate rows.
    indexed = metrics.set_index(["Split", "Mode", "Method"])
    source_ensemble = pd.read_csv(input_root / "ensemble_metrics.csv")
    source_individual = individual.set_index(["Split", "Mode", "Seed"])
    max_replay_difference = 0.0
    for split in SPLITS:
        for mode in MODES + ("MissingMacro",):
            for metric in METRICS:
                saved_pe5 = source_ensemble.loc[
                    source_ensemble.Split.eq(split)
                    & source_ensemble.Mode.eq(mode)
                    & source_ensemble.Metric.eq(metric),
                    "EnsembleValue",
                ]
                if len(saved_pe5) != 1:
                    raise RuntimeError("Saved PE5 metric is missing or duplicated.")
                max_replay_difference = max(
                    max_replay_difference,
                    abs(
                        float(indexed.loc[(split, mode, "PE5"), metric])
                        - float(saved_pe5.iloc[0])
                    ),
                    abs(
                        float(indexed.loc[(split, mode, "Anchor"), metric])
                        - float(source_individual.loc[(split, mode, anchor_seed), metric])
                    ),
                )
    if max_replay_difference > 1e-8:
        raise RuntimeError("Stage 9A Anchor/PE5 metric replay failed.")

    comparison = _full_comparison(metrics)
    retention = _retention_rows(metrics)
    class_gate = _classification_gate(metrics)
    result_class, basic, ratios_pass, direction_consistent = _classification(
        metrics, retention
    )

    metrics.to_csv(output_root / "adpep_metrics.csv", index=False)
    comparison.to_csv(output_root / "adpep_full_comparison.csv", index=False)
    retention.to_csv(output_root / "adpep_retention_analysis.csv", index=False)

    existing_verification = pd.read_csv(
        output_root / "adpep_decision_verification.csv"
    )
    if not existing_verification.Passed.astype(bool).all():
        raise RuntimeError("DECISION PRESERVATION FAILURE")
    if not class_gate.Passed.astype(bool).all():
        raise RuntimeError("DECISION PRESERVATION FAILURE")

    diagnostics = pd.read_csv(output_root / "adpep_boundary_diagnostics.csv")
    fallback = pd.read_csv(output_root / "projection_fallback_samples.csv")
    all_test = retention.loc[
        retention.Split.eq("test") & retention.Variant.eq("ADPEP-All")
    ].set_index("Quantity")
    report = [
        "# Stage 9B ADPEP Final Audit",
        "",
        "## Frozen protocol",
        "",
        "- MainMethod: ADPEP-All",
        "- AblationOnly: ADPEP-57",
        "- Anchor rule: minimum validation J over fixed seeds, lower seed on ties",
        "- AnchorSeed: {}".format(anchor_seed),
        "- Projection labels available: no",
        "- Model training or inference: none",
        "- Stage 9A input hashes unchanged: yes",
        "- Frozen output prediction hashes verified before labels: yes",
        "",
        "## Reproduction and decision gates",
        "",
        "- Maximum Anchor/PE5 metric replay difference: {:.3e}".format(
            max_replay_difference
        ),
        "- ADPEP-All Acc7/Acc5/Acc2/F1 inheritance: exact",
        "- ADPEP-57 Acc7/Acc5 inheritance: exact",
        "- Fallback-to-anchor samples: {}".format(len(fallback)),
        "- Non-finite predictions or metrics: none",
        "",
        "## Main test results",
        "",
        "- Anchor J: {:.12f}".format(
            indexed.loc[("test", "LAV", "Anchor"), "J"]
        ),
        "- PE5 J: {:.12f}".format(indexed.loc[("test", "LAV", "PE5"), "J"]),
        "- ADPEP-All J: {:.12f}".format(
            indexed.loc[("test", "LAV", "ADPEP-All"), "J"]
        ),
        "- Test J retention: {:.6f}".format(
            all_test.loc["J", "RetentionRatio"]
        ),
        "- Test LAV MAE retention: {:.6f}".format(
            all_test.loc["LAV_MAE", "RetentionRatio"]
        ),
        "- Test MissingMacro MAE retention: {:.6f}".format(
            all_test.loc["MissingMacro_MAE", "RetentionRatio"]
        ),
        "- Test LAV Corr gain retention: {:.6f}".format(
            all_test.loc["LAV_Corr", "RetentionRatio"]
        ),
        "- Test MissingMacro Corr gain retention: {:.6f}".format(
            all_test.loc["MissingMacro_Corr", "RetentionRatio"]
        ),
        "",
        "## Success criteria",
        "",
        "- Engineering gates: passed",
        "- Basic performance success: {}".format(basic),
        "- Three required test retention ratios >= 0.50: {}".format(ratios_pass),
        "- Valid/test regression improvement direction consistent: {}".format(
            direction_consistent
        ),
        "- Final classification: **{}**".format(result_class),
        "",
        "The classification is determined mechanically from the frozen criteria; "
        "no test result was used to change the projection rule.",
    ]
    (output_root / "stage9b_adpep_final_audit.md").write_text(
        "\n".join(report) + "\n"
    )

    _verify_input_hashes(input_root, output_root)
    _verify_frozen_outputs(output_root)
    print(
        "Stage9B {}: anchor={} test_J={:.6f} retention={:.3f}".format(
            result_class,
            anchor_seed,
            float(indexed.loc[("test", "LAV", "ADPEP-All"), "J"]),
            float(all_test.loc["J", "RetentionRatio"]),
        )
    )


if __name__ == "__main__":
    main()
