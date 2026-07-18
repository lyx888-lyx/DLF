"""Stage 13A-3: five-fold LOSO CS-DFMRC validation audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stage12a_decision_safe_prototype import (
    METRICS,
    MISSING_MODES,
    aggregate_metrics,
    metrics_for_long,
)
from analysis_stage13a1_safe_oracle import (
    CLASS_METRICS,
    MODES,
    OUTPUT_ROOT,
    SEEDS,
    official_valid_path,
    sha256,
)
from analysis_stage13a2_residual_structure import (
    GROUP_COLUMNS,
    train_residual_frame,
    valid_group_frame,
)
from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
)
from trains.singleTask.cross_seed_median_residual import minimum_group_count
from trains.singleTask.cs_dfmrc_calibrator import (
    build_calibrator,
    calibrator_sha,
    correction_for_group,
    serialize_calibrator,
)


def apply_fold(calibrator, held_valid):
    parts = []
    for mode in MODES:
        selected = held_valid.loc[held_valid.Mode.eq(mode)].sort_values(
            "sample_index", kind="mergesort"
        )
        corrections, sources, weights = [], [], []
        for row in selected.itertuples():
            correction, source, weight = correction_for_group(
                calibrator, row.Mode, row.Cell, row.Half
            )
            corrections.append(correction)
            sources.append(source)
            weights.append(weight)
        baseline = selected.Prediction.to_numpy(np.float32)
        corrections = np.asarray(corrections, dtype=np.float32)
        uncalibrated = baseline + corrections
        calibrated, details = project_array(
            baseline, uncalibrated, "mosi", "adpep_all"
        )
        local = selected.copy()
        local["Correction"] = corrections
        local["CorrectionSource"] = sources
        local["LocalWeight"] = weights
        local["UnprojectedPrediction"] = uncalibrated
        local["CalibratedPrediction"] = calibrated
        local["Projected"] = np.asarray(
            [not detail.pe5_already_feasible for detail in details]
        )
        local["Fallback"] = np.asarray(
            [detail.fallback_to_anchor for detail in details]
        )
        base_decisions = evaluator_decisions(baseline, "mosi")
        calibrated_decisions = evaluator_decisions(calibrated, "mosi")
        if any(
            np.count_nonzero(left != right)
            for left, right in zip(base_decisions, calibrated_decisions)
        ):
            raise RuntimeError("STAGE13_DECISION_PRESERVATION_FAILED")
        parts.append(local)
    return pd.concat(parts, ignore_index=True)


def freeze_loso_predictions(train, train_samples, output):
    n_min = minimum_group_count(train_samples)
    valid = valid_group_frame()
    table_dir = output / "stage13a3_loso_calibration/stage13a3_calibration_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    frames, table_manifest = [], []
    for held_seed in SEEDS:
        training_seeds = [seed for seed in SEEDS if seed != held_seed]
        calibrator = build_calibrator(train, training_seeds, n_min)
        if held_seed in calibrator["TrainingSeeds"]:
            raise RuntimeError("Held-out seed leaked into calibrator.")
        table_path = table_dir / "fold{}_calibrator.json".format(held_seed)
        file_sha = serialize_calibrator(calibrator, table_path)
        if file_sha != sha256(table_path):
            raise RuntimeError("Calibration table SHA mismatch.")
        held_valid = valid.loc[valid.Seed.eq(held_seed)]
        applied = apply_fold(calibrator, held_valid)
        applied["HeldSeed"] = held_seed
        applied["TrainingSeeds"] = ",".join(map(str, training_seeds))
        frames.append(applied)
        table_manifest.append(
            {
                "HeldSeed": held_seed,
                "TrainingSeeds": training_seeds,
                "Path": str(table_path),
                "FileSHA256": file_sha,
                "CanonicalSHA256": calibrator_sha(calibrator),
                "LocalEntries": len(calibrator["LocalEntries"]),
                "PooledEntries": len(calibrator["PooledEntries"]),
                "ContainsHeldSeed": False,
            }
        )
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["HeldSeed", "Mode", "sample_index"], kind="mergesort"
    )
    frozen = predictions[
        [
            "HeldSeed",
            "TrainingSeeds",
            "Split",
            "Mode",
            "sample_index",
            "sample_id",
            "Prediction",
            "Cell",
            "Half",
            "Correction",
            "CorrectionSource",
            "LocalWeight",
            "UnprojectedPrediction",
            "CalibratedPrediction",
            "Projected",
            "Fallback",
        ]
    ]
    path = (
        output
        / "stage13a3_loso_calibration/stage13a3_loso_predictions_valid.csv"
    )
    frozen.to_csv(path, index=False, float_format="%.17g")
    payload = {
        "Path": str(path),
        "SHA256": sha256(path),
        "Rows": len(frozen),
        "LabelsPresent": False,
        "FrozenBeforeValidLabelsRead": True,
        "TestInputsPresent": False,
        "CalibrationTables": table_manifest,
    }
    (
        output
        / "stage13a3_loso_calibration/stage13a3_prediction_manifest.json"
    ).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return pd.read_csv(path, dtype={"sample_id": str}), payload


def load_labels_after_freeze(predictions):
    frames = []
    canonical = None
    for seed in SEEDS:
        labels = pd.read_csv(
            official_valid_path(seed),
            usecols=["sample_index", "sample_id", "label"],
            dtype={"sample_id": str},
        ).sort_values("sample_index", kind="mergesort")
        if canonical is None:
            canonical = labels
        else:
            if not np.array_equal(
                canonical[["sample_index", "sample_id"]].to_numpy(),
                labels[["sample_index", "sample_id"]].to_numpy(),
            ) or not np.allclose(
                canonical.label, labels.label, rtol=0, atol=1e-7
            ):
                raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
        local = predictions.loc[predictions.HeldSeed.eq(seed)].merge(
            labels,
            on=["sample_index", "sample_id"],
            validate="many_to_one",
        )
        if len(local) != len(predictions.loc[predictions.HeldSeed.eq(seed)]):
            raise RuntimeError("STAGE13_SAMPLE_BINDING_FAILED")
        frames.append(local)
    return pd.concat(frames, ignore_index=True)


def evaluate_loso(labeled):
    method_frames = []
    for method, column in (
        ("Baseline", "Prediction"),
        ("Calibrated", "CalibratedPrediction"),
    ):
        local = labeled[
            [
                "HeldSeed",
                "Split",
                "Mode",
                "sample_index",
                "sample_id",
                "label",
            ]
        ].rename(columns={"HeldSeed": "Seed"})
        local = local.copy()
        local["Method"] = method
        local["Prediction"] = labeled[column].to_numpy()
        method_frames.append(local)
    return metrics_for_long(pd.concat(method_frames, ignore_index=True))


def metric_outputs(metrics, output):
    indexed = metrics.set_index(["Seed", "Method", "Mode"])
    rows = []
    for seed in SEEDS:
        for mode in MODES + ("MissingMacro",):
            base = indexed.loc[(seed, "Baseline", mode)]
            cal = indexed.loc[(seed, "Calibrated", mode)]
            row = {"Seed": seed, "Split": "valid", "Mode": mode}
            for metric in ("J",) + METRICS:
                row["Baseline{}".format(metric)] = base[metric]
                row["Calibrated{}".format(metric)] = cal[metric]
                row["Delta{}".format(metric)] = cal[metric] - base[metric]
            rows.append(row)
    per_seed = pd.DataFrame(rows)
    aggregate_rows = []
    numeric = [
        column for column in per_seed.columns if column not in ("Seed", "Split", "Mode")
    ]
    for mode, frame in per_seed.groupby("Mode", sort=True):
        row = {"Mode": mode, "Folds": len(frame)}
        for column in numeric:
            row["{}_Mean".format(column)] = float(frame[column].mean())
            row["{}_Std".format(column)] = float(frame[column].std(ddof=0))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    stage = output / "stage13a3_loso_calibration"
    per_seed.to_csv(stage / "stage13a3_loso_per_seed.csv", index=False)
    aggregate.to_csv(stage / "stage13a3_loso_aggregate.csv", index=False)
    return per_seed, aggregate


def gate_payload(metrics, per_seed, labeled, oracle_per_seed):
    indexed = per_seed.set_index(["Seed", "Mode"])
    delta_j = np.asarray(
        [indexed.loc[(seed, "LAV"), "DeltaJ"] for seed in SEEDS]
    )
    delta_lav = np.asarray(
        [indexed.loc[(seed, "LAV"), "DeltaMAE"] for seed in SEEDS]
    )
    delta_missing = np.asarray(
        [
            indexed.loc[(seed, "MissingMacro"), "DeltaMAE"]
            for seed in SEEDS
        ]
    )
    delta_lav_corr = np.asarray(
        [indexed.loc[(seed, "LAV"), "DeltaCorr"] for seed in SEEDS]
    )
    delta_missing_corr = np.asarray(
        [
            indexed.loc[(seed, "MissingMacro"), "DeltaCorr"]
            for seed in SEEDS
        ]
    )
    remove_best = float(np.delete(delta_j, int(np.argmin(delta_j))).mean())
    base_mean = float(
        per_seed.loc[per_seed.Mode.eq("LAV"), "BaselineJ"].mean()
    )
    cal_mean = float(
        per_seed.loc[per_seed.Mode.eq("LAV"), "CalibratedJ"].mean()
    )
    oracle_mean = float(
        oracle_per_seed.loc[
            oracle_per_seed.Mode.eq("LAV"), "SafeOracleJ"
        ].mean()
    )
    recoverable = (base_mean - cal_mean) / (base_mean - oracle_mean)
    indexed_metrics = metrics.set_index(["Seed", "Method", "Mode"])
    classification = all(
        abs(
            float(indexed_metrics.loc[(seed, "Baseline", mode), metric])
            - float(indexed_metrics.loc[(seed, "Calibrated", mode), metric])
        )
        <= 1e-12
        for seed in SEEDS
        for mode in MODES + ("MissingMacro",)
        for metric in CLASS_METRICS
    )
    mode_delta = {
        mode: float(
            per_seed.loc[per_seed.Mode.eq(mode), "DeltaMAE"].mean()
        )
        for mode in MODES
    }
    missing_improved = sum(mode_delta[mode] < 0 for mode in MISSING_MODES)
    conditions = {
        "MeanDeltaJAtMostMinus0.0015": bool(delta_j.mean() <= -0.0015),
        "AtLeastFourHeldSeedsImproveJ": bool((delta_j < 0).sum() >= 4),
        "WorstSeedDeltaJAtMostPlus0.0005": bool(delta_j.max() <= 0.0005),
        "RemoveBestSeedMeanDeltaJNegative": bool(remove_best < 0),
        "MeanDeltaLAVMAENegative": bool(delta_lav.mean() < 0),
        "MeanDeltaMissingMacroMAENegative": bool(
            delta_missing.mean() < 0
        ),
        "MeanDeltaLAVCorrAtLeastMinus0.0001": bool(
            delta_lav_corr.mean() >= -0.0001
        ),
        "MeanDeltaMissingMacroCorrAtLeastMinus0.0001": bool(
            delta_missing_corr.mean() >= -0.0001
        ),
        "ClassificationInheritedExactly": bool(classification),
        "RecoverableRatioJAtLeast0.15": bool(recoverable >= 0.15),
        "LAVAndAtLeastTwoMissingModesImproveMAE": bool(
            mode_delta["LAV"] < 0 and missing_improved >= 2
        ),
    }
    engineering = {
        "ValidLabelsNotUsedForFitting": True,
        "NoTestAccess": True,
        "ClassificationSampleSignaturesPreserved": True,
        "FallbackZeroOrFloatingBoundaryOnly": bool(
            not labeled.Fallback.any()
        ),
        "CalibrationTablesSerializableAndSHAFrozen": True,
        "NoNaNInf": bool(
            np.isfinite(
                per_seed.select_dtypes(include=[np.number]).to_numpy()
            ).all()
        ),
        "NoHiddenRepresentationInput": True,
    }
    passed = bool(all(engineering.values()) and all(conditions.values()))
    return {
        "Passed": passed,
        "Verdict": (
            "STAGE13A3_LOSO_CALIBRATION_SUPPORTED"
            if passed
            else "STAGE13A3_LOSO_CALIBRATION_UNSUPPORTED"
        ),
        "EngineeringConditions": engineering,
        "Conditions": conditions,
        "Metrics": {
            "MeanDeltaJ": float(delta_j.mean()),
            "ImprovedJSeeds": int((delta_j < 0).sum()),
            "WorstSeedDeltaJ": float(delta_j.max()),
            "RemoveBestSeedMeanDeltaJ": remove_best,
            "MeanDeltaLAVMAE": float(delta_lav.mean()),
            "MeanDeltaMissingMacroMAE": float(delta_missing.mean()),
            "MeanDeltaLAVCorr": float(delta_lav_corr.mean()),
            "MeanDeltaMissingMacroCorr": float(delta_missing_corr.mean()),
            "RecoverableRatioJ": float(recoverable),
            "ModeMeanDeltaMAE": mode_delta,
            "ProjectedRatio": float(labeled.Projected.mean()),
            "ZeroCorrectionRatio": float(
                labeled.CorrectionSource.eq("Zero").mean()
            ),
            "LocalCorrectionRatio": float(
                labeled.CorrectionSource.eq("LocalShrinkage").mean()
            ),
            "PooledCorrectionRatio": float(
                labeled.CorrectionSource.eq("Pooled").mean()
            ),
        },
    }


def main():
    output = OUTPUT_ROOT
    a2_gate = json.loads(
        (
            output
            / "stage13a2_residual_structure/stage13a2_gate.json"
        ).read_text()
    )
    if not a2_gate["Passed"]:
        raise RuntimeError("Stage13A-2 gate is not open.")
    train, train_samples = train_residual_frame()
    predictions, manifest = freeze_loso_predictions(
        train, train_samples, output
    )
    if sha256(Path(manifest["Path"])) != manifest["SHA256"]:
        raise RuntimeError("Frozen LOSO prediction SHA mismatch.")
    labeled = load_labels_after_freeze(predictions)
    metrics = evaluate_loso(labeled)
    per_seed, aggregate = metric_outputs(metrics, output)
    oracle = pd.read_csv(
        output
        / "stage13a1_safe_oracle/stage13a1_safe_oracle_per_seed.csv"
    )
    gate = gate_payload(metrics, per_seed, labeled, oracle)
    stage = output / "stage13a3_loso_calibration"
    (stage / "stage13a3_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    pd.DataFrame(
        [
            {
                "MeanJBaseline": float(
                    per_seed.loc[
                        per_seed.Mode.eq("LAV"), "BaselineJ"
                    ].mean()
                ),
                "MeanJCalibrated": float(
                    per_seed.loc[
                        per_seed.Mode.eq("LAV"), "CalibratedJ"
                    ].mean()
                ),
                "MeanJOracle": float(
                    oracle.loc[
                        oracle.Mode.eq("LAV"), "SafeOracleJ"
                    ].mean()
                ),
                "RecoverableRatioJ": gate["Metrics"]["RecoverableRatioJ"],
            }
        ]
    ).to_csv(stage / "stage13a3_recoverable_ratio.csv", index=False)
    diagnostics = (
        labeled.groupby(["HeldSeed", "CorrectionSource"], sort=True)
        .agg(
            Samples=("sample_index", "size"),
            MeanAbsCorrection=("Correction", lambda x: np.abs(x).mean()),
            ProjectedRatio=("Projected", "mean"),
            FallbackCount=("Fallback", "sum"),
        )
        .reset_index()
    )
    diagnostics.to_csv(
        stage / "stage13a3_correction_diagnostics.csv", index=False
    )
    report = [
        "# Stage 13A-3 LOSO Calibration Audit",
        "",
        "- Five LOSO folds aggregated: true",
        "- Held-out seed train data excluded: true",
        "- Valid labels used for fitting: false",
        "- Test accessed: false",
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
            "- Mean Delta J: {:.9f}".format(
                gate["Metrics"]["MeanDeltaJ"]
            ),
            "- Improved held-out seeds: {}/5".format(
                gate["Metrics"]["ImprovedJSeeds"]
            ),
            "- Worst seed Delta J: {:.9f}".format(
                gate["Metrics"]["WorstSeedDeltaJ"]
            ),
            "- Remove-best-seed Mean Delta J: {:.9f}".format(
                gate["Metrics"]["RemoveBestSeedMeanDeltaJ"]
            ),
            "- RecoverableRatio_J: {:.6f}".format(
                gate["Metrics"]["RecoverableRatioJ"]
            ),
            "- Verdict: **{}**".format(gate["Verdict"]),
            "",
            (
                "Proceed to Stage13B."
                if gate["Passed"]
                else "CS-DFMRC PIPELINE STOPPED BY EVIDENCE GATE"
            ),
        ]
    )
    report_path = stage / "stage13a3_loso_audit.md"
    report_path.write_text("\n".join(report) + "\n")
    stage_manifest = {
        "Stage": "Stage13A-3",
        "Verdict": gate["Verdict"],
        "ValidLabelsUsedForFitting": False,
        "TestLoaderConstructed": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "PredictionSHA256": manifest["SHA256"],
        "Report": str(report_path),
    }
    (stage / "stage13a3_manifest.json").write_text(
        json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(gate["Verdict"], flush=True)
    raise SystemExit(0 if gate["Passed"] else 3)


if __name__ == "__main__":
    main()
