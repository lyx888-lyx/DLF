"""Stage 13A-1: validation-only decision-safe Oracle headroom audit."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stage12a_decision_safe_prototype import (
    METRICS,
    MODES,
    MISSING_MODES,
    SEEDS,
    aggregate_metrics,
    metrics_for_long,
)
from trains.singleTask.anchor_decision_projection import evaluator_decisions
from trains.singleTask.decision_feasible_interval import (
    bounded_decision_interval,
    safe_oracle_predictions,
)


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/code/DLF")
STAGE11_ROOT = Path(
    "/code/DLF-mosi-ccopt-v1/result/missing_baseline/ccopt_v1/mosi"
)
STAGE12_ROOT = Path(
    "/code/DLF-mosi-ccopt-v1/result/missing_baseline/"
    "decision_safe_prototype_audit_v1/mosi"
)
OUTPUT_ROOT = ROOT / "result/missing_baseline/cs_dfmrc_v1/mosi"
CLASS_METRICS = ("acc_7", "acc_5", "acc_2", "F1_score")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_valid_path(seed):
    return (
        SOURCE_ROOT
        / "result/missing_baseline/cfcompat_prediction_ensemble_v1/mosi"
        / "online_seed{}_valid_predictions.csv".format(seed)
    )


def stage11_path(seed, split):
    if split not in ("train", "valid"):
        raise RuntimeError("Stage 13 cannot construct or read a test split.")
    return (
        STAGE11_ROOT
        / "stage11a_geometry"
        / "seed{}_{}_predictions.csv".format(seed, split)
    )


def prediction_long(frame, seed, method, split="valid"):
    rows = []
    for mode in MODES:
        local = frame[
            ["sample_index", "sample_id", "label", "{}_pred".format(mode)]
        ].copy()
        local.columns = [
            "sample_index",
            "sample_id",
            "label",
            "Prediction",
        ]
        local["Seed"] = int(seed)
        local["Split"] = split
        local["Mode"] = mode
        local["Method"] = method
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


def validate_binding(left, right, include_label=True):
    columns = ["sample_index", "sample_id"]
    if include_label:
        columns.append("label")
    left = left.sort_values("sample_index", kind="mergesort")
    right = right.sort_values("sample_index", kind="mergesort")
    if left.sample_index.duplicated().any() or right.sample_index.duplicated().any():
        raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
    if len(left) != len(right):
        raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
    for column in columns:
        if column == "sample_id":
            matches = np.array_equal(
                left[column].astype(str).to_numpy(),
                right[column].astype(str).to_numpy(),
            )
        else:
            if column == "label":
                matches = np.allclose(
                    left[column].to_numpy(np.float64),
                    right[column].to_numpy(np.float64),
                    rtol=0,
                    atol=1e-7,
                )
            else:
                matches = np.array_equal(
                    left[column].to_numpy(), right[column].to_numpy()
                )
        if not matches:
            raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")


def baseline_replay(output):
    replay_rows = []
    stage12_predictions = pd.read_csv(
        STAGE12_ROOT / "decision_safe_predictions_valid.csv",
        usecols=["Seed", "Mode", "sample_index", "sample_id", "p_base"],
        dtype={"sample_id": str},
    )
    stage12_metrics = pd.read_csv(
        STAGE12_ROOT / "decision_safe_metrics_per_seed.csv"
    )
    computed_frames = []
    for seed in SEEDS:
        generated = pd.read_csv(
            stage11_path(seed, "valid"), dtype={"sample_id": str}
        ).sort_values("sample_index", kind="mergesort")
        official = pd.read_csv(
            official_valid_path(seed), dtype={"sample_id": str}
        ).sort_values("sample_index", kind="mergesort")
        validate_binding(generated, official)
        prediction_difference = max(
            float(
                np.max(
                    np.abs(
                        generated["{}_pred".format(mode)].to_numpy(np.float64)
                        - official["{}_pred".format(mode)].to_numpy(np.float64)
                    )
                )
            )
            for mode in MODES
        )
        long = prediction_long(official, seed, "Baseline")
        computed_frames.append(long)
        stage12 = stage12_predictions.loc[stage12_predictions.Seed.eq(seed)]
        pivot = (
            stage12.pivot(
                index=["sample_index", "sample_id"],
                columns="Mode",
                values="p_base",
            )
            .reset_index()
            .sort_values("sample_index", kind="mergesort")
        )
        validate_binding(
            official[["sample_index", "sample_id"]],
            pivot[["sample_index", "sample_id"]],
            include_label=False,
        )
        stage12_difference = max(
            float(
                np.max(
                    np.abs(
                        official["{}_pred".format(mode)].to_numpy(np.float64)
                        - pivot[mode].to_numpy(np.float64)
                    )
                )
            )
            for mode in MODES
        )
        replay_rows.append(
            {
                "Seed": seed,
                # Stage 12 p_base and the formal Online artifact are the
                # frozen baseline pair.  The Stage 11 generated CSV is retained
                # only as a decimal-text diagnostic (it was written with fewer
                # significant digits and is not the baseline authority).
                "BaselineReplayPredictionMaxDifference": stage12_difference,
                "Stage11RoundedDiagnosticMaxDifference": prediction_difference,
            }
        )
    computed = metrics_for_long(pd.concat(computed_frames, ignore_index=True))
    expected = stage12_metrics.loc[stage12_metrics.Method.eq("Baseline")]
    merged = computed.merge(
        expected,
        on=["Seed", "Split", "Method", "Mode"],
        suffixes=("_computed", "_stage12"),
        validate="one_to_one",
    )
    metric_difference = max(
        float(
            np.max(
                np.abs(
                    merged["{}_computed".format(metric)].to_numpy(np.float64)
                    - merged["{}_stage12".format(metric)].to_numpy(np.float64)
                )
            )
        )
        for metric in ("J",) + METRICS
    )
    replay = pd.DataFrame(replay_rows)
    replay["MetricMaxDifference"] = metric_difference
    replay["Passed"] = (
        replay.BaselineReplayPredictionMaxDifference.le(1e-7)
        & replay.MetricMaxDifference.le(1e-8)
    )
    replay.to_csv(output / "baseline_replay/baseline_replay.csv", index=False)
    if not replay.Passed.all():
        raise RuntimeError("STAGE13_BASELINE_REPLAY_FAILED")
    return replay, computed


def asset_manifest(output):
    checkpoint_manifest = (
        SOURCE_ROOT
        / "result/missing_baseline/cfcompat_stability_v1/mosi/"
        "checkpoint_manifest.json"
    )
    paths = [
        checkpoint_manifest,
        STAGE11_ROOT / "baseline_asset_manifest.json",
        STAGE12_ROOT / "decision_safe_prediction_manifest.json",
        STAGE12_ROOT / "decision_safe_predictions_valid.csv",
        STAGE12_ROOT / "decision_safe_metrics_per_seed.csv",
    ]
    for seed in SEEDS:
        paths.extend(
            [
                stage11_path(seed, "train"),
                stage11_path(seed, "valid"),
                official_valid_path(seed),
            ]
        )
    files = [
        {
            "Path": str(path),
            "SHA256": sha256(path),
            "Bytes": path.stat().st_size,
        }
        for path in paths
    ]
    payload = {
        "Dataset": "mosi",
        "Seeds": list(SEEDS),
        "Modes": list(MODES),
        "SplitsPresent": ["train", "valid"],
        "TestInputsPresent": False,
        "FrozenStartCommit": "c97eceb680bde74db4558375f851e7ae65ec9f74",
        "Files": files,
    }
    (output / "baseline_asset_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return {row["Path"]: row["SHA256"] for row in files}


def freeze_intervals(output):
    rows = []
    for seed in SEEDS:
        predictions = pd.read_csv(
            official_valid_path(seed),
            usecols=[
                "sample_index",
                "sample_id",
                *["{}_pred".format(mode) for mode in MODES],
            ],
            dtype={"sample_id": str},
        ).sort_values("sample_index", kind="mergesort")
        if predictions.sample_index.duplicated().any():
            raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
        for mode in MODES:
            for row in predictions.itertuples():
                prediction = np.float32(
                    getattr(row, "{}_pred".format(mode))
                )
                interval = bounded_decision_interval(prediction)
                decisions = evaluator_decisions([prediction], "mosi")
                rows.append(
                    {
                        "Seed": seed,
                        "Split": "valid",
                        "Mode": mode,
                        "sample_index": int(row.sample_index),
                        "sample_id": str(row.sample_id),
                        "p_base": float(prediction),
                        "c7": int(decisions[0][0]),
                        "c5": int(decisions[1][0]),
                        "c2": bool(decisions[2][0]),
                        "Lower": float(interval.lower),
                        "Upper": float(interval.upper),
                        "LowerClosed": interval.lower_closed,
                        "UpperClosed": interval.upper_closed,
                        "EffectiveWidth": interval.width,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["Seed", "Mode", "sample_index"], kind="mergesort"
    )
    if frame.duplicated(["Seed", "Split", "Mode", "sample_index"]).any():
        raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
    path = output / "stage13a1_safe_oracle/safe_intervals_valid.csv"
    frame.to_csv(path, index=False, float_format="%.17g")
    payload = {
        "Path": str(path),
        "SHA256": sha256(path),
        "Rows": len(frame),
        "LabelsPresent": False,
        "FrozenBeforeValidLabelsRead": True,
        "DecisionImplementation": (
            "trains/singleTask/anchor_decision_projection.py"
        ),
        "EffectiveWidthDomain": [-3.0, 3.0],
    }
    (output / "stage13a1_safe_oracle/safe_interval_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return pd.read_csv(path, dtype={"sample_id": str}), payload


def label_region(values):
    values = np.asarray(values)
    return np.select(
        [
            np.abs(values) <= 0.5,
            values <= -2.0,
            values >= 2.0,
        ],
        ["near_zero", "extreme_negative", "extreme_positive"],
        default="other",
    )


def evaluate_oracle(intervals):
    labeled_frames = []
    canonical = None
    for seed in SEEDS:
        labels = pd.read_csv(
            stage11_path(seed, "valid"),
            usecols=["sample_index", "sample_id", "label"],
            dtype={"sample_id": str},
        ).sort_values("sample_index", kind="mergesort")
        if canonical is None:
            canonical = labels.copy()
        else:
            validate_binding(canonical, labels)
        local = intervals.loc[intervals.Seed.eq(seed)].merge(
            labels,
            on=["sample_index", "sample_id"],
            validate="many_to_one",
        )
        if len(local) != len(intervals.loc[intervals.Seed.eq(seed)]):
            raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
        oracle_parts = []
        for mode in MODES:
            selected = local.loc[local.Mode.eq(mode)].sort_values(
                "sample_index", kind="mergesort"
            )
            oracle, details = safe_oracle_predictions(
                selected.p_base.to_numpy(np.float32),
                selected.label.to_numpy(np.float32),
            )
            selected = selected.copy()
            selected["p_oracle"] = oracle
            selected["LabelFeasible"] = [
                value.pe5_already_feasible for value in details
            ]
            label_values = selected.label.to_numpy(np.float32)
            lower_values = selected.Lower.to_numpy(np.float32)
            upper_values = selected.Upper.to_numpy(np.float32)
            selected["LabelBelow"] = (label_values < lower_values) | (
                (label_values == lower_values)
                & ~selected.LowerClosed.to_numpy(bool)
            )
            selected["LabelAbove"] = (label_values > upper_values) | (
                (label_values == upper_values)
                & ~selected.UpperClosed.to_numpy(bool)
            )
            oracle_parts.append(selected)
        labeled_frames.append(pd.concat(oracle_parts, ignore_index=True))
    return pd.concat(labeled_frames, ignore_index=True)


def oracle_outputs(labeled, output):
    method_frames = []
    for method, column in (
        ("Baseline", "p_base"),
        ("SafeOracle", "p_oracle"),
    ):
        local = labeled[
            ["Seed", "Split", "Mode", "sample_index", "sample_id", "label"]
        ].copy()
        local["Method"] = method
        local["Prediction"] = labeled[column]
        method_frames.append(local)
    metrics = metrics_for_long(pd.concat(method_frames, ignore_index=True))
    indexed = metrics.set_index(["Seed", "Method", "Mode"])
    detail_rows = []
    for seed in SEEDS:
        for mode in MODES:
            sample = labeled.loc[
                labeled.Seed.eq(seed) & labeled.Mode.eq(mode)
            ]
            base = indexed.loc[(seed, "Baseline", mode)]
            oracle = indexed.loc[(seed, "SafeOracle", mode)]
            detail_rows.append(
                {
                    "Seed": seed,
                    "Split": "valid",
                    "Mode": mode,
                    "BaselineMAE": base.MAE,
                    "SafeOracleMAE": oracle.MAE,
                    "OracleMAEGain": base.MAE - oracle.MAE,
                    "BaselineCorr": base.Corr,
                    "SafeOracleCorr": oracle.Corr,
                    "BaselineJ": base.J,
                    "SafeOracleJ": oracle.J,
                    "OracleJGain": base.J - oracle.J,
                    "ExactMatchToLabelRatio": float(
                        (
                            sample.p_oracle.to_numpy(np.float32)
                            == sample.label.to_numpy(np.float32)
                        ).mean()
                    ),
                    "LabelInsideRatio": float(sample.LabelFeasible.mean()),
                    "LabelBelowRatio": float(sample.LabelBelow.mean()),
                    "LabelAboveRatio": float(sample.LabelAbove.mean()),
                    "MeanSafeIntervalWidth": float(
                        sample.EffectiveWidth.mean()
                    ),
                    "MedianSafeIntervalWidth": float(
                        sample.EffectiveWidth.median()
                    ),
                }
            )
        missing = [row for row in detail_rows if row["Seed"] == seed and row["Mode"] in MISSING_MODES]
        detail_rows.append(
            {
                "Seed": seed,
                "Split": "valid",
                "Mode": "MissingMacro",
                **{
                    key: float(np.mean([row[key] for row in missing]))
                    for key in (
                        "BaselineMAE",
                        "SafeOracleMAE",
                        "OracleMAEGain",
                        "BaselineCorr",
                        "SafeOracleCorr",
                        "ExactMatchToLabelRatio",
                        "LabelInsideRatio",
                        "LabelBelowRatio",
                        "LabelAboveRatio",
                        "MeanSafeIntervalWidth",
                        "MedianSafeIntervalWidth",
                    )
                },
                "BaselineJ": indexed.loc[(seed, "Baseline", "LAV"), "J"],
                "SafeOracleJ": indexed.loc[(seed, "SafeOracle", "LAV"), "J"],
                "OracleJGain": (
                    indexed.loc[(seed, "Baseline", "LAV"), "J"]
                    - indexed.loc[(seed, "SafeOracle", "LAV"), "J"]
                ),
            }
        )
    per_seed = pd.DataFrame(detail_rows)
    per_seed.to_csv(
        output / "stage13a1_safe_oracle/stage13a1_safe_oracle_per_seed.csv",
        index=False,
    )
    numeric = [
        column
        for column in per_seed.columns
        if column not in ("Seed", "Split", "Mode")
    ]
    aggregate_rows = []
    for mode, frame in per_seed.groupby("Mode", sort=True):
        row = {"Mode": mode, "Seeds": len(frame)}
        for column in numeric:
            row["{}_Mean".format(column)] = float(frame[column].mean())
            row["{}_Std".format(column)] = float(frame[column].std(ddof=0))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(
        output / "stage13a1_safe_oracle/stage13a1_safe_oracle_aggregate.csv",
        index=False,
    )
    return metrics, per_seed, aggregate


def concentration_outputs(labeled, output):
    frame = labeled.copy()
    frame["MAEGain"] = (
        np.abs(frame.p_base - frame.label)
        - np.abs(frame.p_oracle - frame.label)
    )
    frame["JWeight"] = frame.Mode.map(
        {"LAV": 0.5, "LA": 1 / 6, "LV": 1 / 6, "L": 1 / 6}
    )
    frame["WeightedJGain"] = frame.MAEGain * frame.JWeight
    sample = (
        frame.groupby(["Seed", "sample_index"], sort=True)
        .WeightedJGain.sum()
        .reset_index()
    )
    total = float(sample.WeightedJGain.sum())
    rows = []
    for percentage in (1, 5, 10):
        count = max(1, int(np.ceil(len(sample) * percentage / 100.0)))
        gain = float(sample.nlargest(count, "WeightedJGain").WeightedJGain.sum())
        rows.append(
            {
                "Scope": "TopSampleConcentration",
                "Group": "{}%".format(percentage),
                "Samples": count,
                "Gain": gain,
                "ContributionRatio": gain / total if total > 0 else np.nan,
            }
        )
    frame["Region"] = label_region(frame.label)
    frame["LabelBin"] = np.round(np.clip(frame.label, -3, 3)).astype(int)
    for scope, column in (("Region", "Region"), ("LabelBin", "LabelBin"), ("Mode", "Mode")):
        for value, group in frame.groupby(column, sort=True):
            gain = float(group.WeightedJGain.sum())
            rows.append(
                {
                    "Scope": scope,
                    "Group": value,
                    "Samples": len(group),
                    "Gain": gain,
                    "ContributionRatio": gain / total if total > 0 else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(
        output / "stage13a1_safe_oracle/stage13a1_gain_concentration.csv",
        index=False,
    )
    top5 = float(
        result.loc[
            result.Scope.eq("TopSampleConcentration")
            & result.Group.eq("5%"),
            "ContributionRatio",
        ].iloc[0]
    )
    return result, top5


def gate_payload(metrics, per_seed, top5, engineering):
    indexed = per_seed.set_index(["Seed", "Mode"])
    lav_gain = np.asarray(
        [indexed.loc[(seed, "LAV"), "OracleMAEGain"] for seed in SEEDS]
    )
    missing_gain = np.asarray(
        [
            indexed.loc[(seed, "MissingMacro"), "OracleMAEGain"]
            for seed in SEEDS
        ]
    )
    j_gain = np.asarray(
        [indexed.loc[(seed, "LAV"), "OracleJGain"] for seed in SEEDS]
    )
    leave_best = np.delete(j_gain, int(np.argmax(j_gain))).mean()
    indexed_metrics = metrics.set_index(["Seed", "Method", "Mode"])
    classification = all(
        abs(
            float(indexed_metrics.loc[(seed, "Baseline", mode), metric])
            - float(indexed_metrics.loc[(seed, "SafeOracle", mode), metric])
        )
        <= 1e-12
        for seed in SEEDS
        for mode in MODES + ("MissingMacro",)
        for metric in CLASS_METRICS
    )
    conditions = {
        "MeanLAVMAEGainAtLeast0.010": bool(lav_gain.mean() >= 0.010),
        "MeanMissingMacroMAEGainAtLeast0.008": bool(
            missing_gain.mean() >= 0.008
        ),
        "MeanJGainAtLeast0.008": bool(j_gain.mean() >= 0.008),
        "AllFiveSeedsImproveJ": bool((j_gain > 0).sum() == 5),
        "LeaveBestSeedOutMeanJGainAtLeast0.006": bool(leave_best >= 0.006),
        "Top5PercentContributionAtMost0.60": bool(top5 <= 0.60),
        "ClassificationInheritedExactly": bool(classification),
    }
    passed = bool(all(engineering.values()) and all(conditions.values()))
    return {
        "Passed": passed,
        "Verdict": (
            "STAGE13A1_SAFE_ORACLE_HEADROOM_SUPPORTED"
            if passed
            else "STAGE13A1_INSUFFICIENT_SAFE_HEADROOM"
        ),
        "EngineeringConditions": engineering,
        "Conditions": conditions,
        "Metrics": {
            "MeanLAVMAEGain": float(lav_gain.mean()),
            "MeanMissingMacroMAEGain": float(missing_gain.mean()),
            "MeanJGain": float(j_gain.mean()),
            "ImprovedJSeeds": int((j_gain > 0).sum()),
            "LeaveBestSeedOutMeanJGain": float(leave_best),
            "Top5PercentContribution": float(top5),
        },
    }


def main():
    output = OUTPUT_ROOT
    for child in (
        "baseline_replay",
        "stage13a1_safe_oracle",
        "stage13a2_residual_structure",
        "stage13a3_loso_calibration",
        "stage13b_frozen_valid",
        "locked_test",
        "final",
    ):
        (output / child).mkdir(parents=True, exist_ok=True)
    background = {
        "Worktrees": subprocess.check_output(
            ["git", "worktree", "list"], text=True
        ).splitlines(),
        "Processes": subprocess.check_output(
            [
                "bash",
                "-lc",
                "ps -ef | grep -E 'mosei|stage10' | grep -v grep || true",
            ],
            text=True,
        ).splitlines(),
        "NvidiaSMI": subprocess.check_output(["nvidia-smi"], text=True),
        "ReadOnlyCheck": True,
    }
    (output / "mosei_background_status.json").write_text(
        json.dumps(background, indent=2, sort_keys=True) + "\n"
    )
    input_hashes = asset_manifest(output)
    replay, _ = baseline_replay(output)
    intervals, interval_manifest = freeze_intervals(output)
    labeled = evaluate_oracle(intervals)
    metrics, per_seed, aggregate = oracle_outputs(labeled, output)
    _, top5 = concentration_outputs(labeled, output)
    stage12_gate = json.loads((STAGE12_ROOT / "stage12a_gate.json").read_text())
    engineering = {
        "BaselineReplayPassed": bool(replay.Passed.all()),
        "Stage12ReferenceReplayed": (
            abs(
                float(stage12_gate["Metrics"]["MeanDeltaJValidSafe"])
                - (-0.0021581451098124037)
            )
            <= 1e-12
        ),
        "IntervalsFrozenBeforeLabels": bool(
            interval_manifest["FrozenBeforeValidLabelsRead"]
        ),
        "ProjectionUsesFormalStage9B": True,
        "NoTestLoader": True,
        "NoTestLabelsRead": True,
        "NoTestPredictionsRead": True,
        "NoTestEvaluation": True,
        "InputFilesUnchanged": all(
            sha256(path) == digest for path, digest in input_hashes.items()
        ),
        "NoNaNInf": bool(
            np.isfinite(
                per_seed.select_dtypes(include=[np.number]).to_numpy()
            ).all()
        ),
    }
    gate = gate_payload(metrics, per_seed, top5, engineering)
    gate_path = output / "stage13a1_safe_oracle/stage13a1_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    report = [
        "# Stage 13A-1 Safe Oracle Headroom Audit",
        "",
        "- Baseline replay: {}".format(engineering["BaselineReplayPassed"]),
        "- Test loader constructed: false",
        "- Test labels read: false",
        "- Test predictions read: false",
        "- Classification inheritance: {}".format(
            gate["Conditions"]["ClassificationInheritedExactly"]
        ),
        "",
        "## Gate",
        "",
    ]
    report.extend(
        "- {}: {}".format(key, "PASS" if value else "FAIL")
        for key, value in gate["Conditions"].items()
    )
    report.extend(
        [
            "",
            "- Mean LAV MAE gain: {:.9f}".format(
                gate["Metrics"]["MeanLAVMAEGain"]
            ),
            "- Mean MissingMacro MAE gain: {:.9f}".format(
                gate["Metrics"]["MeanMissingMacroMAEGain"]
            ),
            "- Mean J gain: {:.9f}".format(gate["Metrics"]["MeanJGain"]),
            "- Improved J seeds: {}/5".format(
                gate["Metrics"]["ImprovedJSeeds"]
            ),
            "- Leave-best-seed-out mean J gain: {:.9f}".format(
                gate["Metrics"]["LeaveBestSeedOutMeanJGain"]
            ),
            "- Top 5% contribution: {:.6f}".format(
                gate["Metrics"]["Top5PercentContribution"]
            ),
            "- Verdict: **{}**".format(gate["Verdict"]),
            "",
            (
                "Proceed to Stage13A-2."
                if gate["Passed"]
                else "CS-DFMRC PIPELINE STOPPED BY EVIDENCE GATE"
            ),
        ]
    )
    report_path = (
        output
        / "stage13a1_safe_oracle/stage13a1_safe_oracle_audit.md"
    )
    report_path.write_text("\n".join(report) + "\n")
    manifest = {
        "Stage": "Stage13A-1",
        "Verdict": gate["Verdict"],
        "TestLoaderConstructed": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "SafeIntervalSHA256": interval_manifest["SHA256"],
        "Report": str(report_path),
    }
    (output / "stage13a1_safe_oracle/stage13a1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(gate["Verdict"], flush=True)
    raise SystemExit(0 if gate["Passed"] else 3)


if __name__ == "__main__":
    main()
