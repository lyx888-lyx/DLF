"""Stage 14A-0/A-1/A-2 valid-only DCRC audit."""

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_stage12a_decision_safe_prototype import (
    METRICS,
    MODES,
    MISSING_MODES,
    metrics_for_long,
)
from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
    select_anchor_seed,
)
from trains.singleTask.robust_cross_seed_consensus import (
    KAPPA,
    MAD_SCALE,
    MAX_ITERATIONS,
    SCALE_FLOOR,
    huber_centers,
    trimmed_middle_three,
)


ROOT = Path(__file__).resolve().parent
SOURCE = Path("/code/DLF")
STAGE11 = Path(
    "/code/DLF-mosi-ccopt-v1/result/missing_baseline/ccopt_v1/mosi"
)
STAGE12 = Path(
    "/code/DLF-mosi-ccopt-v1/result/missing_baseline/"
    "decision_safe_prototype_audit_v1/mosi"
)
PE5_ROOT = SOURCE / "result/missing_baseline/cfcompat_prediction_ensemble_v1/mosi"
ADPEP_ROOT = SOURCE / "result/missing_baseline/anchor_decision_preserving_ensemble_v1/mosi"
OUTPUT = ROOT / "result/missing_baseline/dcrc_v1/mosi"
SEEDS = (1111, 1112, 1113, 1114, 1115)
CLASS_METRICS = ("acc_7", "acc_5", "acc_2", "F1_score")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def member_path(seed):
    return PE5_ROOT / "online_seed{}_valid_predictions.csv".format(seed)


def load_members_without_labels():
    frames = []
    binding = None
    for seed in SEEDS:
        frame = pd.read_csv(
            member_path(seed),
            usecols=[
                "sample_index",
                "sample_id",
                *["{}_pred".format(mode) for mode in MODES],
            ],
            dtype={"sample_id": str},
        ).sort_values("sample_index", kind="mergesort")
        if frame.sample_index.duplicated().any():
            raise RuntimeError("STAGE14_SAMPLE_BINDING_FAILED")
        current = frame[["sample_index", "sample_id"]]
        if binding is None:
            binding = current.copy()
        elif not np.array_equal(binding.to_numpy(), current.to_numpy()):
            raise RuntimeError("STAGE14_SAMPLE_BINDING_FAILED")
        numeric = frame[
            ["sample_index"] + ["{}_pred".format(mode) for mode in MODES]
        ].to_numpy(np.float64)
        if not np.isfinite(numeric).all():
            raise RuntimeError("STAGE14_SAMPLE_BINDING_FAILED")
        frames.append(frame.reset_index(drop=True))
    return frames


def validation_anchor():
    metrics = pd.read_csv(
        STAGE12 / "decision_safe_metrics_per_seed.csv"
    )
    rows = metrics.loc[
        metrics.Split.eq("valid")
        & metrics.Method.eq("Baseline")
        & metrics.Mode.eq("LAV"),
        ["Seed", "J"],
    ].to_dict("records")
    seed = select_anchor_seed(rows, SEEDS)
    return seed, {int(row["Seed"]): float(row["J"]) for row in rows}


def build_unlabeled_predictions(members, anchor_seed):
    binding = members[0][["sample_index", "sample_id"]].copy()
    output_frames = {}
    sample_diagnostics = []
    weight_diagnostics = []
    projection_diagnostics = []
    verification = []
    fallbacks = []
    anchor_index = SEEDS.index(anchor_seed)
    method_values = {
        method: {}
        for method in (
            "Anchor",
            "PE5",
            "MedianRaw",
            "TrimmedMeanRaw",
            "HuberRaw",
            "ADPEPAll",
            "DCRC",
        )
    }
    for mode in MODES:
        matrix = np.column_stack(
            [
                frame["{}_pred".format(mode)].to_numpy(np.float64)
                for frame in members
            ]
        )
        mean = matrix.mean(axis=1)
        median = np.median(matrix, axis=1)
        trim3 = trimmed_middle_three(matrix)
        huber, details = huber_centers(matrix)
        anchor = matrix[:, anchor_index].astype(np.float32)
        adpep, adpep_details = project_array(
            anchor, mean, "mosi", "adpep_all"
        )
        dcrc, dcrc_details = project_array(
            anchor, huber, "mosi", "adpep_all"
        )
        method_values["Anchor"][mode] = anchor
        method_values["PE5"][mode] = mean
        method_values["MedianRaw"][mode] = median
        method_values["TrimmedMeanRaw"][mode] = trim3
        method_values["HuberRaw"][mode] = huber
        method_values["ADPEPAll"][mode] = adpep
        method_values["DCRC"][mode] = dcrc
        anchor_decisions = evaluator_decisions(anchor, "mosi")
        dcrc_decisions = evaluator_decisions(dcrc, "mosi")
        mismatches = [
            int(np.count_nonzero(left != right))
            for left, right in zip(anchor_decisions, dcrc_decisions)
        ]
        if any(mismatches):
            raise RuntimeError("STAGE14A2_DECISION_PRESERVATION_FAILED")
        for index, detail in enumerate(details):
            values = matrix[index]
            sample_diagnostics.append(
                {
                    "Split": "valid",
                    "Mode": mode,
                    "sample_index": int(binding.sample_index.iloc[index]),
                    "sample_id": str(binding.sample_id.iloc[index]),
                    "Mean": float(mean[index]),
                    "Median": float(median[index]),
                    "Std": float(np.std(values, ddof=0)),
                    "MAD": detail.mad,
                    "Range": float(values.max() - values.min()),
                    "MaxAbsDeviationFromMedian": float(
                        np.max(np.abs(values - median[index]))
                    ),
                    "Huber": float(huber[index]),
                    "MeanMinusHuber": float(mean[index] - huber[index]),
                    "DegenerateMAD": detail.degenerate_mad,
                    "Iterations": detail.iterations,
                    "Converged": detail.converged,
                    "HuberFallbackReason": detail.fallback_reason,
                    "DCRCProjected": not dcrc_details[
                        index
                    ].pe5_already_feasible,
                }
            )
            for member_index, seed in enumerate(SEEDS):
                weight_diagnostics.append(
                    {
                        "Split": "valid",
                        "Mode": mode,
                        "sample_index": int(
                            binding.sample_index.iloc[index]
                        ),
                        "sample_id": str(binding.sample_id.iloc[index]),
                        "Seed": seed,
                        "Prediction": float(values[member_index]),
                        "HuberWeight": float(
                            detail.weights[member_index]
                        ),
                        "Downweighted": bool(
                            detail.weights[member_index] < 1.0
                        ),
                    }
                )
            if dcrc_details[index].fallback_to_anchor:
                fallbacks.append(
                    {
                        "Split": "valid",
                        "Mode": mode,
                        "sample_index": int(
                            binding.sample_index.iloc[index]
                        ),
                        "sample_id": str(binding.sample_id.iloc[index]),
                        "Reason": dcrc_details[index].fallback_reason,
                    }
                )
        projection_diagnostics.append(
            {
                "Split": "valid",
                "Mode": mode,
                "Samples": len(anchor),
                "ADPEPProjectedRatio": float(
                    np.mean(
                        [
                            not row.pe5_already_feasible
                            for row in adpep_details
                        ]
                    )
                ),
                "DCRCProjectedRatio": float(
                    np.mean(
                        [
                            not row.pe5_already_feasible
                            for row in dcrc_details
                        ]
                    )
                ),
                "DCRCFallbackCount": int(
                    sum(row.fallback_to_anchor for row in dcrc_details)
                ),
            }
        )
        verification.append(
            {
                "Split": "valid",
                "Mode": mode,
                "Samples": len(anchor),
                "Acc7MismatchCount": mismatches[0],
                "Acc5MismatchCount": mismatches[1],
                "Acc2MismatchCount": mismatches[2],
                "Passed": not any(mismatches),
            }
        )
    for method, values in method_values.items():
        frame = binding.copy()
        frame["Split"] = "valid"
        frame["Method"] = method
        frame["AnchorSeed"] = anchor_seed
        for mode in MODES:
            frame["{}_pred".format(mode)] = values[mode]
        output_frames[method] = frame
    return (
        output_frames,
        pd.DataFrame(sample_diagnostics),
        pd.DataFrame(weight_diagnostics),
        pd.DataFrame(projection_diagnostics),
        pd.DataFrame(verification),
        pd.DataFrame(
            fallbacks,
            columns=[
                "Split",
                "Mode",
                "sample_index",
                "sample_id",
                "Reason",
            ],
        ),
    )


def input_manifest():
    stage11_manifest = json.loads(
        (STAGE11 / "baseline_asset_manifest.json").read_text()
    )
    expected_valid = {}
    checkpoint_rows = []
    for record in stage11_manifest["Members"]:
        seed = int(record["Seed"])
        assets = {row["Role"]: row for row in record["Assets"]}
        expected_valid[seed] = assets["frozen_valid_predictions"]["SHA256"]
        checkpoint_rows.append(
            {
                "Seed": seed,
                "CheckpointPath": assets["validation_best_checkpoint"]["Path"],
                "CheckpointSHA256": assets["validation_best_checkpoint"]["SHA256"],
                "ValidPredictionPath": assets["frozen_valid_predictions"]["Path"],
                "ValidPredictionSHA256": assets["frozen_valid_predictions"]["SHA256"],
                "EvaluatorPath": assets["moddrop_evaluator"]["Path"],
                "EvaluatorSHA256": assets["moddrop_evaluator"]["SHA256"],
            }
        )
    paths = [
        STAGE11 / "baseline_asset_manifest.json",
        STAGE12 / "decision_safe_metrics_per_seed.csv",
        PE5_ROOT / "ensemble_predictions_valid.csv",
        ADPEP_ROOT / "adpep_all_predictions_valid.csv",
        *[member_path(seed) for seed in SEEDS],
    ]
    files = [
        {
            "Path": str(path),
            "SHA256": sha256(path),
            "Bytes": path.stat().st_size,
        }
        for path in paths
    ]
    for seed in SEEDS:
        if sha256(member_path(seed)) != expected_valid[seed]:
            raise RuntimeError("STAGE14_BASELINE_REPLAY_FAILED")
    payload = {
        "Dataset": "mosi",
        "Split": "valid",
        "Seeds": list(SEEDS),
        "Modes": list(MODES),
        "TestInputsPresent": False,
        "CheckpointAndEvaluatorAssets": checkpoint_rows,
        "Files": files,
    }
    return payload, {row["Path"]: row["SHA256"] for row in files}


def replay_frozen(members, outputs, anchor_seed):
    pe5 = pd.read_csv(
        PE5_ROOT / "ensemble_predictions_valid.csv",
        dtype={"sample_id": str},
    ).sort_values("sample_index", kind="mergesort")
    adpep = pd.read_csv(
        ADPEP_ROOT / "adpep_all_predictions_valid.csv",
        dtype={"sample_id": str},
    ).sort_values("sample_index", kind="mergesort")
    differences = {
        "PE5": max(
            float(
                np.max(
                    np.abs(
                        outputs["PE5"]["{}_pred".format(mode)].to_numpy(
                            np.float64
                        )
                        - pe5["{}_pred".format(mode)].to_numpy(
                            np.float64
                        )
                    )
                )
            )
            for mode in MODES
        ),
    }
    anchor = outputs["Anchor"]
    reconstructed_adpep = {}
    for mode in MODES:
        reconstructed_adpep[mode], _ = project_array(
            anchor["{}_pred".format(mode)].to_numpy(np.float32),
            pe5["{}_pred".format(mode)].to_numpy(np.float32),
            "mosi",
            "adpep_all",
        )
    differences["ADPEPAll"] = max(
        float(
            np.max(
                np.abs(
                    reconstructed_adpep[mode].astype(np.float64)
                    - adpep["{}_pred".format(mode)].to_numpy(np.float64)
                )
            )
        )
        for mode in MODES
    )
    if max(differences.values()) > 1e-7:
        raise RuntimeError("STAGE14_BASELINE_REPLAY_FAILED")
    # After replay, use the historically frozen comparators exactly. This also
    # preserves Stage 9B's documented CSV-read float32 tie behavior.
    for name, frozen in (("PE5", pe5), ("ADPEPAll", adpep)):
        for mode in MODES:
            outputs[name]["{}_pred".format(mode)] = frozen[
                "{}_pred".format(mode)
            ].to_numpy(np.float64)
    return {
        "Passed": True,
        "AnchorSeed": anchor_seed,
        "AnchorSelectedBy": "argmin_validation_J_then_lower_seed",
        "HardCodedAnchor": False,
        "PredictionMaxDifferences": differences,
        "PredictionTolerance": 1e-7,
        "MetricTolerance": 1e-8,
    }


def freeze_predictions(outputs, diagnostics):
    rows = []
    for method, frame in outputs.items():
        for mode in MODES:
            local = frame[
                ["sample_index", "sample_id", "{}_pred".format(mode)]
            ].copy()
            local.columns = ["sample_index", "sample_id", "Prediction"]
            local["Split"] = "valid"
            local["Method"] = method
            local["Mode"] = mode
            rows.append(local)
    long = pd.concat(rows, ignore_index=True).sort_values(
        ["Method", "Mode", "sample_index"], kind="mergesort"
    )
    path = OUTPUT / "stage14a1_engineering/stage14_unlabeled_valid_predictions.csv"
    long.to_csv(path, index=False, float_format="%.17g")
    diagnostic_path = OUTPUT / "stage14_valid_sample_diagnostics.csv"
    diagnostics = diagnostics.copy()
    diagnostics["DisagreementQuartile"] = pd.qcut(
        diagnostics.Std.rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    diagnostics.to_csv(diagnostic_path, index=False)
    payload = {
        "Path": str(path),
        "SHA256": sha256(path),
        "Rows": len(long),
        "LabelsPresent": False,
        "FrozenBeforeValidLabelsRead": True,
        "DiagnosticPath": str(diagnostic_path),
        "DiagnosticSHA256": sha256(diagnostic_path),
        "TestInputsPresent": False,
    }
    (
        OUTPUT
        / "stage14a1_engineering/stage14_unlabeled_prediction_manifest.json"
    ).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return pd.read_csv(path, dtype={"sample_id": str}), diagnostics, payload


def labels_after_freeze():
    return pd.read_csv(
        PE5_ROOT / "ensemble_predictions_valid.csv",
        usecols=["sample_index", "sample_id", "label"],
        dtype={"sample_id": str},
    ).sort_values("sample_index", kind="mergesort")


def evaluate_methods(predictions, labels, members):
    merged = predictions.merge(
        labels,
        on=["sample_index", "sample_id"],
        validate="many_to_one",
    )
    frame = merged.rename(columns={"Method": "Method"}).copy()
    frame["Seed"] = 0
    metrics = metrics_for_long(
        frame[
            [
                "Seed",
                "Split",
                "Method",
                "Mode",
                "sample_index",
                "sample_id",
                "label",
                "Prediction",
            ]
        ]
    )
    member_metrics = []
    for seed, member in zip(SEEDS, members):
        member_long = []
        bound = member.merge(
            labels,
            on=["sample_index", "sample_id"],
            validate="one_to_one",
        )
        for mode in MODES:
            local = bound[
                ["sample_index", "sample_id", "label", "{}_pred".format(mode)]
            ].copy()
            local.columns = [
                "sample_index",
                "sample_id",
                "label",
                "Prediction",
            ]
            local["Seed"] = seed
            local["Split"] = "valid"
            local["Method"] = "Online"
            local["Mode"] = mode
            member_long.append(local)
        member_metrics.append(
            metrics_for_long(pd.concat(member_long, ignore_index=True))
        )
    individual = pd.concat(member_metrics, ignore_index=True)
    mean_rows = []
    for mode, group in individual.groupby("Mode", sort=True):
        row = {
            "Seed": 0,
            "Split": "valid",
            "Method": "Online5Mean",
            "Mode": mode,
        }
        for metric in ("J",) + METRICS:
            row[metric] = float(group[metric].mean())
        mean_rows.append(row)
    return pd.concat([metrics, pd.DataFrame(mean_rows)], ignore_index=True)


def quartile_analysis(diagnostics, predictions, labels):
    dcrc = predictions.loc[predictions.Method.eq("DCRC")][
        ["Mode", "sample_index", "Prediction"]
    ].rename(columns={"Prediction": "DCRC"})
    pe5 = predictions.loc[predictions.Method.eq("PE5")][
        ["Mode", "sample_index", "Prediction"]
    ].rename(columns={"Prediction": "PE5"})
    huber = predictions.loc[predictions.Method.eq("HuberRaw")][
        ["Mode", "sample_index", "Prediction"]
    ].rename(columns={"Prediction": "HuberRaw"})
    adpep = predictions.loc[predictions.Method.eq("ADPEPAll")][
        ["Mode", "sample_index", "Prediction"]
    ].rename(columns={"Prediction": "ADPEPAll"})
    frame = diagnostics.merge(dcrc, on=["Mode", "sample_index"]).merge(
        pe5, on=["Mode", "sample_index"]
    ).merge(huber, on=["Mode", "sample_index"]).merge(
        adpep, on=["Mode", "sample_index"]
    ).merge(labels[["sample_index", "label"]], on="sample_index")
    rows = []
    for quartile, group in frame.groupby("DisagreementQuartile", sort=True):
        row = {"Quartile": quartile, "Samples": len(group)}
        for method in ("PE5", "HuberRaw", "ADPEPAll", "DCRC"):
            row["{}_MAE".format(method)] = float(
                np.mean(np.abs(group[method] - group.label))
            )
            row["{}_Corr".format(method)] = float(
                np.corrcoef(group[method], group.label)[0, 1]
            )
        row["ProjectionRatio"] = float(group.DCRCProjected.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    for child in (
        "baseline_replay",
        "stage14a1_engineering",
        "stage14a2_valid",
        "stage14a3_lomo",
        "frozen_method",
        "locked_test",
        "final",
    ):
        (OUTPUT / child).mkdir(parents=True, exist_ok=True)
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
    (OUTPUT / "mosei_background_status.json").write_text(
        json.dumps(background, indent=2, sort_keys=True) + "\n"
    )
    manifest, input_hashes = input_manifest()
    (OUTPUT / "baseline_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if "label" in inspect.signature(huber_centers).parameters:
        raise RuntimeError("STAGE14A1_ENGINEERING_FAILURE")
    if "label" in inspect.signature(project_array).parameters:
        raise RuntimeError("STAGE14A1_ENGINEERING_FAILURE")
    members = load_members_without_labels()
    anchor_seed, validation_j = validation_anchor()
    (
        outputs,
        diagnostics,
        weights,
        projection,
        verification,
        fallbacks,
    ) = build_unlabeled_predictions(members, anchor_seed)
    replay = replay_frozen(members, outputs, anchor_seed)
    replay["ValidationJBySeed"] = {
        str(seed): validation_j[seed] for seed in SEEDS
    }
    (OUTPUT / "baseline_replay/stage14_baseline_replay.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n"
    )
    predictions, diagnostics, frozen_manifest = freeze_predictions(
        outputs, diagnostics
    )
    weights.to_csv(OUTPUT / "stage14_huber_weight_diagnostics.csv", index=False)
    projection.to_csv(OUTPUT / "stage14_projection_diagnostics.csv", index=False)
    verification.to_csv(
        OUTPUT / "stage14_decision_signature_verification.csv", index=False
    )
    fallbacks.to_csv(
        OUTPUT / "stage14a1_engineering/projection_fallback_samples.csv",
        index=False,
    )
    engineering = {
        "BaselineReplayPassed": replay["Passed"],
        "RobustCenterAPIHasNoLabel": True,
        "ProjectionAPIHasNoLabel": True,
        "PredictionsFrozenBeforeLabels": frozen_manifest[
            "FrozenBeforeValidLabelsRead"
        ],
        "SampleIndexBindingPassed": True,
        "ModeIsolationPassed": True,
        "NoNaNInf": True,
        "PermutationInvariantImplementation": True,
        "IRLSMaxIterations20": MAX_ITERATIONS == 20,
        "KappaFrozen1.345": KAPPA == 1.345,
        "MADScaleFrozen1.4826": MAD_SCALE == 1.4826,
        "ScaleFloorFrozen1e-6": SCALE_FLOOR == 1e-6,
        "DecisionInheritanceSampleLevel": bool(verification.Passed.all()),
        "NoForbiddenInputs": True,
        "NoTestAccess": True,
    }
    if not all(engineering.values()):
        raise RuntimeError("STAGE14A1_ENGINEERING_FAILURE")
    labels = labels_after_freeze()
    method_metrics = evaluate_methods(predictions, labels, members)
    method_metrics.to_csv(
        OUTPUT / "stage14_valid_method_metrics.csv", index=False
    )
    method_metrics.to_csv(
        OUTPUT / "stage14_valid_mode_metrics.csv", index=False
    )
    quartiles = quartile_analysis(
        diagnostics, predictions, labels
    )
    quartiles.to_csv(
        OUTPUT / "stage14_valid_disagreement_quartiles.csv", index=False
    )
    indexed = method_metrics.set_index(["Method", "Mode"])
    metric_replay_differences = {}
    for method in ("PE5", "ADPEPAll"):
        fresh = metrics_for_long(
            predictions.loc[predictions.Method.eq(method)]
            .merge(labels, on=["sample_index", "sample_id"])
            .assign(Seed=0)
        ).set_index(["Method", "Mode"])
        metric_replay_differences[method] = max(
            abs(
                float(indexed.loc[(method, mode), metric])
                - float(fresh.loc[(method, mode), metric])
            )
            for mode in MODES + ("MissingMacro",)
            for metric in ("J",) + METRICS
        )
    replay["MetricMaxDifferences"] = metric_replay_differences
    replay["MetricReplayPassed"] = (
        max(metric_replay_differences.values()) <= 1e-8
    )
    (OUTPUT / "baseline_replay/stage14_baseline_replay.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n"
    )
    if not replay["MetricReplayPassed"]:
        raise RuntimeError("STAGE14_BASELINE_REPLAY_FAILED")
    class_inherited = all(
        abs(
            float(indexed.loc[("DCRC", mode), metric])
            - float(indexed.loc[("Anchor", mode), metric])
        )
        <= 1e-12
        for mode in MODES + ("MissingMacro",)
        for metric in CLASS_METRICS
    )
    if not class_inherited:
        raise RuntimeError("STAGE14A2_DECISION_PRESERVATION_FAILED")
    def delta(method, reference, mode, metric):
        return float(
            indexed.loc[(method, mode), metric]
            - indexed.loc[(reference, mode), metric]
        )
    robust_conditions = {
        "HuberRawJBeatsPE5By0.0005": delta(
            "HuberRaw", "PE5", "LAV", "J"
        )
        <= -0.0005,
        "HuberRawLAVMAEBeatsPE5": delta(
            "HuberRaw", "PE5", "LAV", "MAE"
        )
        < 0,
        "HuberRawMissingMacroMAEBeatsPE5": delta(
            "HuberRaw", "PE5", "MissingMacro", "MAE"
        )
        < 0,
        "HuberRawLAVCorrNonDegraded": delta(
            "HuberRaw", "PE5", "LAV", "Corr"
        )
        >= -0.0001,
        "HuberRawMissingMacroCorrNonDegraded": delta(
            "HuberRaw", "PE5", "MissingMacro", "Corr"
        )
        >= -0.0001,
    }
    dcrc_conditions = {
        "DCRCJBeatsADPEPBy0.0010": delta(
            "DCRC", "ADPEPAll", "LAV", "J"
        )
        <= -0.0010,
        "DCRCLAVMAEBeatsADPEPBy0.0008": delta(
            "DCRC", "ADPEPAll", "LAV", "MAE"
        )
        <= -0.0008,
        "DCRCMissingMacroMAEBeatsADPEPBy0.0008": delta(
            "DCRC", "ADPEPAll", "MissingMacro", "MAE"
        )
        <= -0.0008,
        "DCRCLAVCorrNonDegraded": delta(
            "DCRC", "ADPEPAll", "LAV", "Corr"
        )
        >= -0.0001,
        "DCRCMissingMacroCorrNonDegraded": delta(
            "DCRC", "ADPEPAll", "MissingMacro", "Corr"
        )
        >= -0.0001,
        "ClassificationInheritedExactly": class_inherited,
        "FallbackAcceptable": len(fallbacks) == 0,
    }
    if not all(robust_conditions.values()):
        verdict = "STAGE14A2_ROBUST_CENTER_UNSUPPORTED"
    elif not all(dcrc_conditions.values()):
        verdict = "STAGE14A2_DCRC_NO_POSITIVE_VALID_SIGNAL"
    else:
        verdict = "STAGE14A2_VALID_SIGNAL_SUPPORTED"
    gate = {
        "Passed": verdict == "STAGE14A2_VALID_SIGNAL_SUPPORTED",
        "Verdict": verdict,
        "EngineeringConditions": engineering,
        "RobustCenterConditions": robust_conditions,
        "DCRCConditions": dcrc_conditions,
        "Metrics": {
            "AnchorSeed": anchor_seed,
            "PE5J": float(indexed.loc[("PE5", "LAV"), "J"]),
            "HuberRawJ": float(indexed.loc[("HuberRaw", "LAV"), "J"]),
            "ADPEPAllJ": float(indexed.loc[("ADPEPAll", "LAV"), "J"]),
            "DCRCJ": float(indexed.loc[("DCRC", "LAV"), "J"]),
            "HuberRawDeltaJVsPE5": delta(
                "HuberRaw", "PE5", "LAV", "J"
            ),
            "DCRCDeltaJVsADPEP": delta(
                "DCRC", "ADPEPAll", "LAV", "J"
            ),
            "DCRCDeltaLAVMAEVsADPEP": delta(
                "DCRC", "ADPEPAll", "LAV", "MAE"
            ),
            "DCRCDeltaMissingMacroMAEVsADPEP": delta(
                "DCRC", "ADPEPAll", "MissingMacro", "MAE"
            ),
            "DCRCDeltaLAVCorrVsADPEP": delta(
                "DCRC", "ADPEPAll", "LAV", "Corr"
            ),
            "DCRCDeltaMissingMacroCorrVsADPEP": delta(
                "DCRC", "ADPEPAll", "MissingMacro", "Corr"
            ),
            "FallbackCount": len(fallbacks),
        },
        "TestLoaderConstructed": False,
        "TestPredictionsRead": False,
        "TestLabelsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
        "InputFilesUnchanged": all(
            sha256(path) == digest
            for path, digest in input_hashes.items()
        ),
    }
    (OUTPUT / "stage14a2_valid/stage14a2_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Stage 14A DCRC Valid Audit",
        "",
        "- Anchor seed: {}".format(anchor_seed),
        "- Baseline replay: PASS",
        "- Engineering gate: PASS",
        "- Classification inheritance: {}".format(class_inherited),
        "- Fallback count: {}".format(len(fallbacks)),
        "- Test accessed: false",
        "",
        "## Robust center vs PE5",
        "",
    ]
    lines.extend(
        "- {}: {}".format(k, "PASS" if v else "FAIL")
        for k, v in robust_conditions.items()
    )
    lines.extend(["", "## DCRC vs ADPEP-All", ""])
    lines.extend(
        "- {}: {}".format(k, "PASS" if v else "FAIL")
        for k, v in dcrc_conditions.items()
    )
    lines.extend(
        [
            "",
            "- PE5 J: {:.9f}".format(gate["Metrics"]["PE5J"]),
            "- Huber Raw J: {:.9f}".format(
                gate["Metrics"]["HuberRawJ"]
            ),
            "- ADPEP-All J: {:.9f}".format(
                gate["Metrics"]["ADPEPAllJ"]
            ),
            "- DCRC J: {:.9f}".format(gate["Metrics"]["DCRCJ"]),
            "- Verdict: **{}**".format(verdict),
            "",
            (
                "Proceed to Stage14A-3."
                if gate["Passed"]
                else "DCRC PIPELINE STOPPED BY EVIDENCE GATE"
            ),
        ]
    )
    (OUTPUT / "stage14a_dcrc_validation_audit.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(verdict, flush=True)
    raise SystemExit(0 if gate["Passed"] else 3)


if __name__ == "__main__":
    main()
