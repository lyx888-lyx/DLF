"""Stage 12A validation-only, label-free decision-safe prototype audit."""
import argparse
import hashlib
import inspect
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_stage11a_prototype_geometry import (
    MODES,
    MISSING_MODES,
    SOURCE_ROOT,
    load_model,
    model_args,
    online_records,
)
from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
)
from trains.singleTask.missing_utils import mode_to_mask, regression_metrics
from trains.singleTask.ordered_prototype_geometry import (
    FusionRepresentationCapture,
    acc7_bins,
    prototype_estimate,
)
from utils.functions import setup_seed

SEEDS = (1111, 1112, 1113, 1114, 1115)
STAGE11_ROOT = Path(
    "result/missing_baseline/ccopt_v1/mosi"
)
OUTPUT_ROOT = Path(
    "result/missing_baseline/decision_safe_prototype_audit_v1/mosi"
)
METRICS = ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE", "Loss")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_prototype_correction(baseline, prototype_estimate_values):
    """The Stage 11A step is intentionally fixed and has no tunable argument."""
    # Stage 11A evaluated the fixed correction in NumPy float64 because its
    # prototype estimate is float64.  Preserve that arithmetic exactly so the
    # raw arm is a replay, not a numerically distinct reimplementation.
    baseline = np.asarray(baseline, dtype=np.float64)
    estimate = np.asarray(prototype_estimate_values, dtype=np.float64)
    if baseline.shape != estimate.shape:
        raise ValueError("Baseline/prototype estimate shapes differ.")
    return np.clip(baseline + 0.10 * (estimate - baseline), -3.0, 3.0)


def retention(baseline, raw, safe, higher_is_better=False):
    if higher_is_better:
        raw_gain = float(raw) - float(baseline)
        safe_gain = float(safe) - float(baseline)
    else:
        raw_gain = float(baseline) - float(raw)
        safe_gain = float(baseline) - float(safe)
    return raw_gain, safe_gain, (
        safe_gain / raw_gain if raw_gain > 0 else None
    )


class UnlabeledValidDataset(Dataset):
    """Access valid features and IDs without indexing the label field."""

    def __init__(self, path):
        with Path(path).open("rb") as handle:
            data = pickle.load(handle)["valid"]
        self.text = data["text_bert"].astype(np.float32)
        self.audio = data["audio"].astype(np.float32)
        self.vision = data["vision"].astype(np.float32)
        self.ids = data["id"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {
            "text": torch.tensor(self.text[index]),
            "audio": torch.tensor(self.audio[index]),
            "vision": torch.tensor(self.vision[index]),
            "sample_index": int(index),
            "sample_id": self.ids[index],
        }


def extract_unlabeled_valid(model, args, num_workers):
    dataset = UnlabeledValidDataset(args.featurePath)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    capture = FusionRepresentationCapture(model)
    hidden = {mode: [] for mode in MODES}
    predictions = {mode: [] for mode in MODES}
    sample_indices, sample_ids = [], []
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            sample_indices.extend(batch["sample_index"].view(-1).tolist())
            sample_ids.extend(str(value) for value in list(batch["sample_id"]))
            for mode in MODES:
                mask = mode_to_mask(
                    mode,
                    len(batch["sample_index"]),
                    args.device,
                    audio.dtype,
                )
                output = model(text, audio, vision, mask)
                hidden[mode].append(capture.pop().detach().cpu())
                predictions[mode].append(
                    output["output_logit"].detach().view(-1).cpu()
                )
    capture.close()
    return {
        "sample_index": np.asarray(sample_indices, dtype=np.int64),
        "sample_id": np.asarray(sample_ids, dtype=object),
        "hidden": {
            mode: torch.cat(values).numpy() for mode, values in hidden.items()
        },
        "model_predictions": {
            mode: torch.cat(values).numpy()
            for mode, values in predictions.items()
        },
    }


def stage11_files(stage11_root):
    geometry = stage11_root / "stage11a_geometry"
    paths = [
        stage11_root / "baseline_asset_manifest.json",
        geometry / "stage11a_geometry_per_seed.csv",
        geometry / "stage11a_prototype_metrics.csv",
        geometry / "stage11a_direction_diagnostic.csv",
        geometry / "stage11a_baseline_replay.csv",
        geometry / "stage11a_gate.json",
        geometry / "stage11a_geometry_audit.md",
    ]
    paths.extend(
        geometry / "seed{}_valid_predictions.csv".format(seed)
        for seed in SEEDS
    )
    return paths


def input_manifest(paths):
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "Path": str(path),
                "SHA256": sha256(path),
                "Bytes": path.stat().st_size,
            }
        )
    return {
        "Dataset": "mosi",
        "Split": "valid",
        "Seeds": list(SEEDS),
        "Modes": list(MODES),
        "Files": rows,
        "TestInputsPresent": False,
    }


def load_prototypes(prototype_frame, seed, mode):
    selected = prototype_frame.loc[
        prototype_frame.Seed.eq(seed) & prototype_frame.Mode.eq(mode)
    ].sort_values("Level")
    if len(selected) != 7:
        raise RuntimeError("Prototype rows are incomplete.")
    result = {}
    for row in selected.itertuples():
        result[int(row.Level)] = (
            None
            if bool(row.EmptyBin)
            else np.asarray(json.loads(row.PrototypeVector), dtype=np.float64)
        )
    return result


def validate_unlabeled_binding(baseline, extracted, seed):
    if baseline.sample_index.duplicated().any():
        raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
    if not np.array_equal(
        baseline.sample_index.to_numpy(np.int64),
        extracted["sample_index"],
    ):
        raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
    if not np.array_equal(
        baseline.sample_id.astype(str).to_numpy(),
        extracted["sample_id"].astype(str),
    ):
        raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
    numeric = baseline[
        ["sample_index"] + ["{}_pred".format(mode) for mode in MODES]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")


def metrics_for_long(frame):
    rows = []
    for (seed, method), group in frame.groupby(["Seed", "Method"], sort=True):
        mode_metrics = {}
        for mode in MODES:
            selected = group.loc[group.Mode.eq(mode)].sort_values(
                "sample_index", kind="mergesort"
            )
            values = regression_metrics(
                torch.tensor(selected.Prediction.to_numpy(), dtype=torch.float32),
                torch.tensor(selected.label.to_numpy(), dtype=torch.float32),
            )
            values["Loss"] = values["MAE"]
            mode_metrics[mode] = values
        mode_metrics["MissingMacro"] = {
            metric: float(
                np.mean(
                    [mode_metrics[mode][metric] for mode in MISSING_MODES]
                )
            )
            for metric in METRICS
        }
        j_value = 0.5 * mode_metrics["LAV"]["MAE"] + 0.5 * mode_metrics[
            "MissingMacro"
        ]["MAE"]
        for mode in MODES + ("MissingMacro",):
            rows.append(
                {
                    "Seed": int(seed),
                    "Split": "valid",
                    "Method": method,
                    "Mode": mode,
                    "J": j_value,
                    **mode_metrics[mode],
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(metrics):
    numeric = ["J"] + list(METRICS)
    rows = []
    for (method, mode), frame in metrics.groupby(["Method", "Mode"], sort=True):
        row = {"Method": method, "Mode": mode, "Seeds": len(frame)}
        for metric in numeric:
            row["{}_Mean".format(metric)] = float(frame[metric].mean())
            row["{}_Std".format(metric)] = float(frame[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def retention_rows(metrics):
    indexed = metrics.set_index(["Seed", "Method", "Mode"])
    rows = []
    quantities = (
        ("J", "LAV", "J", False),
        ("LAV_MAE", "LAV", "MAE", False),
        ("MissingMacro_MAE", "MissingMacro", "MAE", False),
        ("LAV_Corr", "LAV", "Corr", True),
        ("MissingMacro_Corr", "MissingMacro", "Corr", True),
    )
    for seed in SEEDS:
        for quantity, mode, metric, higher in quantities:
            base = indexed.loc[(seed, "Baseline", mode), metric]
            raw = indexed.loc[(seed, "Raw", mode), metric]
            safe = indexed.loc[(seed, "Safe", mode), metric]
            raw_gain, safe_gain, ratio = retention(base, raw, safe, higher)
            rows.append(
                {
                    "Scope": "Seed",
                    "Seed": seed,
                    "Quantity": quantity,
                    "Baseline": base,
                    "Raw": raw,
                    "Safe": safe,
                    "RawGain": raw_gain,
                    "SafeGain": safe_gain,
                    "Retention": ratio,
                }
            )
    for quantity, mode, metric, higher in quantities:
        values = {}
        for method in ("Baseline", "Raw", "Safe"):
            values[method] = float(
                metrics.loc[
                    metrics.Method.eq(method) & metrics.Mode.eq(mode), metric
                ].mean()
            )
        raw_gain, safe_gain, ratio = retention(
            values["Baseline"], values["Raw"], values["Safe"], higher
        )
        rows.append(
            {
                "Scope": "Aggregate",
                "Seed": np.nan,
                "Quantity": quantity,
                **values,
                "RawGain": raw_gain,
                "SafeGain": safe_gain,
                "Retention": ratio,
            }
        )
    return pd.DataFrame(rows)


def stage12_gate(metrics, retention_frame, engineering):
    indexed = metrics.set_index(["Seed", "Method", "Mode"])
    delta_j = np.asarray(
        [
            indexed.loc[(seed, "Safe", "LAV"), "J"]
            - indexed.loc[(seed, "Baseline", "LAV"), "J"]
            for seed in SEEDS
        ]
    )
    delta_lav_mae = np.asarray(
        [
            indexed.loc[(seed, "Safe", "LAV"), "MAE"]
            - indexed.loc[(seed, "Baseline", "LAV"), "MAE"]
            for seed in SEEDS
        ]
    )
    delta_missing_mae = np.asarray(
        [
            indexed.loc[(seed, "Safe", "MissingMacro"), "MAE"]
            - indexed.loc[(seed, "Baseline", "MissingMacro"), "MAE"]
            for seed in SEEDS
        ]
    )
    delta_lav_corr = np.asarray(
        [
            indexed.loc[(seed, "Safe", "LAV"), "Corr"]
            - indexed.loc[(seed, "Baseline", "LAV"), "Corr"]
            for seed in SEEDS
        ]
    )
    delta_missing_corr = np.asarray(
        [
            indexed.loc[(seed, "Safe", "MissingMacro"), "Corr"]
            - indexed.loc[(seed, "Baseline", "MissingMacro"), "Corr"]
            for seed in SEEDS
        ]
    )
    aggregate_retention = retention_frame.loc[
        retention_frame.Scope.eq("Aggregate")
        & retention_frame.Quantity.eq("J"),
        "Retention",
    ].iloc[0]
    largest_gain_index = int(np.argmin(delta_j))
    leave_one_out = np.delete(delta_j, largest_gain_index).mean()
    regression = {
        "MeanDeltaJAtMostMinus0.00125": bool(delta_j.mean() <= -0.00125),
        "AggregateRetentionJAtLeast0.50": bool(
            np.isfinite(aggregate_retention) and aggregate_retention >= 0.50
        ),
        "AtLeast4SeedsImproveJ": bool((delta_j < 0).sum() >= 4),
        "WorstSeedDeltaJAtMostPlus0.0005": bool(delta_j.max() <= 0.0005),
        "MeanDeltaLAVMAENegative": bool(delta_lav_mae.mean() < 0),
        "MeanDeltaMissingMacroMAENegative": bool(
            delta_missing_mae.mean() < 0
        ),
        "MeanDeltaLAVCorrAtLeastMinus0.0001": bool(
            delta_lav_corr.mean() >= -0.0001
        ),
        "MeanDeltaMissingMacroCorrAtLeastMinus0.0001": bool(
            delta_missing_corr.mean() >= -0.0001
        ),
        "LeaveLargestGainOutMeanDeltaJNegative": bool(leave_one_out < 0),
    }
    all_engineering = bool(all(engineering.values()))
    passed = all_engineering and all(regression.values())
    if not all_engineering:
        classification = "STAGE12A_ENGINEERING_FAILURE"
    elif passed:
        classification = (
            "STAGE12A_DECISION_SAFE_PROTOTYPE_SIGNAL_SUPPORTED"
        )
    elif delta_j.mean() < 0:
        classification = "STAGE12A_WEAK_POSITIVE_SIGNAL"
    else:
        classification = "STAGE12A_NO_REGRESSION_SIGNAL"
    return {
        "Passed": bool(passed),
        "Verdict": classification,
        "EngineeringConditions": engineering,
        "RegressionConditions": regression,
        "Metrics": {
            "MeanDeltaJValidSafe": float(delta_j.mean()),
            "AggregateRetentionJ": float(aggregate_retention),
            "ImprovedJSeeds": int((delta_j < 0).sum()),
            "WorstSeedDeltaJ": float(delta_j.max()),
            "MeanDeltaLAVMAE": float(delta_lav_mae.mean()),
            "MeanDeltaMissingMacroMAE": float(delta_missing_mae.mean()),
            "MeanDeltaLAVCorr": float(delta_lav_corr.mean()),
            "MeanDeltaMissingMacroCorr": float(delta_missing_corr.mean()),
            "LeaveLargestGainOutMeanDeltaJ": float(leave_one_out),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage11-root", default=str(STAGE11_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--num-workers", type=int, default=1)
    cli = parser.parse_args()
    stage11_root = Path(cli.stage11_root)
    output = Path(cli.output_root)
    output.mkdir(parents=True, exist_ok=True)

    pre_status = subprocess.check_output(
        [
            "bash",
            "-lc",
            "git status --short --branch && git diff --stat && git diff --check",
        ],
        text=True,
    )
    (output / "pre_audit_git_status.txt").write_text(pre_status)
    stage10_state_path = Path(
        "/code/DLF-mosei-generalization-v1/runtime/mosei_generalization_v1/state.json"
    )
    stage10_state = json.loads(stage10_state_path.read_text())
    background = {
        "Stage10State": stage10_state,
        "Stage10StatePath": str(stage10_state_path),
        "Stage10StateSHA256": sha256(stage10_state_path),
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

    source_paths = stage11_files(stage11_root)
    inputs = input_manifest(source_paths)
    input_hashes = {row["Path"]: row["SHA256"] for row in inputs["Files"]}
    (output / "input_manifest_with_sha.json").write_text(
        json.dumps(inputs, indent=2, sort_keys=True) + "\n"
    )
    stage11_gate = json.loads(
        (stage11_root / "stage11a_geometry/stage11a_gate.json").read_text()
    )
    known = {
        "DeltaJ": -0.0024780591328938803,
        "DeltaLAVMAE": -0.0018171668052673,
        "DeltaMissingMacroMAE": -0.00313895146052042,
        "DeltaMeanAcc7": -0.006986899563318739,
        "DeltaMeanAcc5": -0.00458515283842792,
    }
    replay_differences = {
        key: abs(float(stage11_gate["Means"][key]) - value)
        for key, value in known.items()
    }
    initial_replay = max(replay_differences.values()) <= 1e-6
    if not initial_replay:
        raise RuntimeError("STAGE12A_STAGE11A_REPLAY_FAILED")

    if "label" in inspect.signature(project_array).parameters:
        raise RuntimeError("Projection API accepts forbidden labels.")
    prototype_frame = pd.read_csv(
        stage11_root / "stage11a_geometry/stage11a_prototype_metrics.csv"
    )
    geometry_frame = pd.read_csv(
        stage11_root / "stage11a_geometry/stage11a_geometry_per_seed.csv"
    )
    records, _, _ = online_records()
    prediction_rows = []
    boundary_rows = []
    verification_rows = []
    fallback_rows = []
    stage11_reconstructed = []
    extracted_prediction_max = []
    for record in records:
        seed = int(record["Seed"])
        setup_seed(seed)
        args = model_args(seed)
        model, _ = load_model(args, record)
        extracted = extract_unlabeled_valid(model, args, cli.num_workers)
        baseline_path = (
            stage11_root
            / "stage11a_geometry"
            / "seed{}_valid_predictions.csv".format(seed)
        )
        baseline = pd.read_csv(
            baseline_path,
            usecols=[
                "sample_index",
                "sample_id",
                *["{}_pred".format(mode) for mode in MODES],
            ],
        ).sort_values("sample_index", kind="mergesort")
        validate_unlabeled_binding(baseline, extracted, seed)
        for mode in MODES:
            model_difference = float(
                np.max(
                    np.abs(
                        extracted["model_predictions"][mode].astype(np.float64)
                        - baseline["{}_pred".format(mode)].to_numpy(np.float64)
                    )
                )
            )
            extracted_prediction_max.append(model_difference)
            # The persisted Stage 11A CSV is decimal text; compare a fresh
            # deterministic extraction using the protocol's 1e-6 replay
            # tolerance rather than requiring binary identity after parsing.
            if model_difference > 1e-6:
                raise RuntimeError("STAGE12A_STAGE11A_REPLAY_FAILED")
            prototypes = load_prototypes(prototype_frame, seed, mode)
            temperature = float(
                geometry_frame.loc[
                    geometry_frame.Seed.eq(seed)
                    & geometry_frame.Mode.eq(mode),
                    "PrototypeTemperature",
                ].iloc[0]
            )
            q_proto = prototype_estimate(
                extracted["hidden"][mode], prototypes, temperature
            )
            p_base = baseline["{}_pred".format(mode)].to_numpy(np.float32)
            p_raw = raw_prototype_correction(p_base, q_proto)
            p_safe, details = project_array(
                p_base, p_raw, "mosi", "adpep_all"
            )
            projector_raw = np.asarray(p_raw, dtype=np.float32)
            base_decisions = evaluator_decisions(p_base, "mosi")
            raw_decisions = evaluator_decisions(p_raw, "mosi")
            safe_decisions = evaluator_decisions(p_safe, "mosi")
            mismatches = [
                int(np.count_nonzero(left != right))
                for left, right in zip(base_decisions, safe_decisions)
            ]
            if any(mismatches):
                raise RuntimeError("STAGE12A_DECISION_PRESERVATION_FAILED")
            constraints = np.column_stack(
                [
                    left != right
                    for left, right in zip(base_decisions, raw_decisions)
                ]
            )
            projected = np.asarray(
                [not value.pe5_already_feasible for value in details],
                dtype=bool,
            )
            boundary_rows.append(
                {
                    "Seed": seed,
                    "Split": "valid",
                    "Mode": mode,
                    "TotalSamples": len(p_safe),
                    "RawAlreadySafeCount": int((~constraints.any(axis=1)).sum()),
                    "RawAlreadySafeRatio": float(
                        (~constraints.any(axis=1)).mean()
                    ),
                    "ProjectedCount": int(projected.sum()),
                    "ProjectedRatio": float(projected.mean()),
                    "SafeEqualsRawRatio": float(
                        (p_safe == projector_raw).mean()
                    ),
                    "SafeEqualsBaseRatio": float((p_safe == p_base).mean()),
                    "BoundaryAdjustedCount": int(
                        sum(value.boundary_adjusted for value in details)
                    ),
                    "FallbackCount": int(
                        sum(value.fallback_to_anchor for value in details)
                    ),
                    "MeanAbsRawMinusBase": float(
                        np.mean(np.abs(p_raw - p_base))
                    ),
                    "MeanAbsSafeMinusBase": float(
                        np.mean(np.abs(p_safe - p_base))
                    ),
                    "MeanAbsSafeMinusRaw": float(
                        np.mean(np.abs(p_safe - projector_raw))
                    ),
                    "MaxAbsSafeMinusRaw": float(
                        np.max(np.abs(p_safe - projector_raw))
                    ),
                    "OnlyAcc7LimitedRatio": float(
                        ((constraints[:, 0]) & (constraints.sum(axis=1) == 1)).mean()
                    ),
                    "OnlyAcc5LimitedRatio": float(
                        ((constraints[:, 1]) & (constraints.sum(axis=1) == 1)).mean()
                    ),
                    "OnlyAcc2LimitedRatio": float(
                        ((constraints[:, 2]) & (constraints.sum(axis=1) == 1)).mean()
                    ),
                    "MultipleBoundaryLimitedRatio": float(
                        (constraints.sum(axis=1) > 1).mean()
                    ),
                }
            )
            verification_rows.append(
                {
                    "Seed": seed,
                    "Split": "valid",
                    "Mode": mode,
                    "Acc7MismatchCount": mismatches[0],
                    "Acc5MismatchCount": mismatches[1],
                    "Acc2MismatchCount": mismatches[2],
                    "Passed": not any(mismatches),
                }
            )
            for index, detail in enumerate(details):
                if detail.fallback_to_anchor:
                    fallback_rows.append(
                        {
                            "Seed": seed,
                            "Split": "valid",
                            "Mode": mode,
                            "sample_index": int(
                                baseline.sample_index.iloc[index]
                            ),
                            "sample_id": str(
                                baseline.sample_id.iloc[index]
                            ),
                            "Reason": detail.fallback_reason,
                        }
                    )
                prediction_rows.append(
                    {
                        "Seed": seed,
                        "Split": "valid",
                        "Mode": mode,
                        "sample_index": int(
                            baseline.sample_index.iloc[index]
                        ),
                        "sample_id": str(baseline.sample_id.iloc[index]),
                        "p_base": float(p_base[index]),
                        "q_proto": float(q_proto[index]),
                        "p_raw": float(p_raw[index]),
                        "p_safe": float(p_safe[index]),
                    }
                )
        del model
        torch.cuda.empty_cache()
        print("Stage12A unlabeled seed{} complete".format(seed), flush=True)

    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["Seed", "Mode", "sample_index"], kind="mergesort"
    )
    if predictions.duplicated(["Seed", "Split", "Mode", "sample_index"]).any():
        raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
    numeric = predictions[
        ["sample_index", "p_base", "q_proto", "p_raw", "p_safe"]
    ].to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
    frozen_path = output / "decision_safe_predictions_valid.csv"
    predictions.to_csv(frozen_path, index=False, float_format="%.17g")
    frozen_sha = sha256(frozen_path)
    prediction_manifest = {
        "Path": str(frozen_path),
        "SHA256": frozen_sha,
        "Rows": len(predictions),
        "LabelsPresent": False,
        "ProjectionAPIAcceptsLabel": False,
        "PredictionsFrozenBeforeLabelsLoaded": True,
        "Dataset": "mosi",
        "Split": "valid",
        "Seeds": list(SEEDS),
        "Modes": list(MODES),
        "RawStep": 0.10,
        "ProjectionComparisonPrecision": "float32",
    }
    (output / "decision_safe_prediction_manifest.json").write_text(
        json.dumps(prediction_manifest, indent=2, sort_keys=True) + "\n"
    )
    if sha256(frozen_path) != frozen_sha:
        raise RuntimeError("Frozen p_safe SHA mismatch.")
    # All labeled evaluation below consumes the immutable on-disk artifact,
    # rather than the pre-freeze in-memory frame.
    predictions = pd.read_csv(frozen_path, dtype={"sample_id": str})
    if len(predictions) != prediction_manifest["Rows"]:
        raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
    if "label" in predictions.columns:
        raise RuntimeError("Frozen prediction artifact unexpectedly has labels.")

    # Labels are loaded only after the label-free predictions and SHA are frozen.
    labeled_frames = []
    canonical_binding = None
    for seed in SEEDS:
        source = (
            stage11_root
            / "stage11a_geometry"
            / "seed{}_valid_predictions.csv".format(seed)
        )
        labels = pd.read_csv(
            source,
            usecols=["sample_index", "sample_id", "label"],
            dtype={"sample_id": str},
        ).sort_values("sample_index", kind="mergesort")
        if labels.sample_index.duplicated().any():
            raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
        binding = labels[["sample_index", "sample_id", "label"]].copy()
        binding["sample_id"] = binding.sample_id.astype(str)
        if canonical_binding is None:
            canonical_binding = binding
        else:
            if not np.array_equal(
                canonical_binding.to_numpy(), binding.to_numpy()
            ):
                raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
        seed_predictions = predictions.loc[predictions.Seed.eq(seed)]
        merged = seed_predictions.merge(
            labels,
            on=["sample_index", "sample_id"],
            validate="many_to_one",
        )
        if len(merged) != len(seed_predictions):
            raise RuntimeError("STAGE12A_SAMPLE_BINDING_FAILED")
        labeled_frames.append(merged)
    labeled = pd.concat(labeled_frames, ignore_index=True)
    method_frames = []
    for method, column in (
        ("Baseline", "p_base"),
        ("Raw", "p_raw"),
        ("Safe", "p_safe"),
    ):
        local = labeled[
            ["Seed", "Split", "Mode", "sample_index", "sample_id", "label"]
        ].copy()
        local["Method"] = method
        local["Prediction"] = labeled[column]
        method_frames.append(local)
    evaluation = pd.concat(method_frames, ignore_index=True)
    metrics = metrics_for_long(evaluation)
    aggregate = aggregate_metrics(metrics)
    retention_frame = retention_rows(metrics)

    stage11_diagnostic = pd.read_csv(
        stage11_root / "stage11a_geometry/stage11a_direction_diagnostic.csv"
    )
    indexed = metrics.set_index(["Seed", "Method", "Mode"])
    replay_checks = []
    for row in stage11_diagnostic.itertuples():
        raw = indexed.loc[(row.Seed, "Raw", row.Mode)]
        base = indexed.loc[(row.Seed, "Baseline", row.Mode)]
        differences = {
            "JBaseline": abs(float(base.J) - float(row.JBaseline)),
            "JDiagnostic": abs(float(raw.J) - float(row.JDiagnostic)),
            "DeltaJ": abs(
                float(raw.J - base.J) - float(row.DeltaJ)
            ),
            "BaselineMAE": abs(
                float(base.MAE) - float(row.BaselineMAE)
            ),
            "DiagnosticMAE": abs(
                float(raw.MAE) - float(row.DiagnosticMAE)
            ),
        }
        replay_checks.append(max(differences.values()))
    reconstructed_replay = max(replay_checks) <= 1e-6
    replay_payload = {
        "KnownAggregateDifferences": replay_differences,
        "KnownAggregateReplayPassed": initial_replay,
        "ReconstructedMetricMaxDifference": max(replay_checks),
        "ReconstructedReplayPassed": reconstructed_replay,
        "ModelPredictionMaxDifference": max(extracted_prediction_max),
        "ModelPredictionTolerance": 1e-6,
        "Tolerance": 1e-6,
        "Passed": initial_replay and reconstructed_replay,
    }
    (output / "stage11a_replay_check.json").write_text(
        json.dumps(replay_payload, indent=2, sort_keys=True) + "\n"
    )
    if not replay_payload["Passed"]:
        raise RuntimeError("STAGE12A_STAGE11A_REPLAY_FAILED")

    verification = pd.DataFrame(verification_rows)
    boundary = pd.DataFrame(boundary_rows)
    class_metrics = ("acc_7", "acc_5", "acc_2", "F1_score")
    for seed in SEEDS:
        for mode in MODES + ("MissingMacro",):
            base = indexed.loc[(seed, "Baseline", mode)]
            safe = indexed.loc[(seed, "Safe", mode)]
            if any(
                abs(float(base[metric]) - float(safe[metric])) > 1e-12
                for metric in class_metrics
            ):
                raise RuntimeError("STAGE12A_DECISION_PRESERVATION_FAILED")

    label_diagnostics = []
    labeled["LabelLevel"] = acc7_bins(labeled.label.to_numpy())
    labeled["BaselineLevel"] = evaluator_decisions(
        labeled.p_base.to_numpy(np.float32), "mosi"
    )[0]
    labeled["Projected"] = (
        labeled.p_safe.to_numpy(np.float32)
        != labeled.p_raw.to_numpy(np.float32)
    )
    labeled["AbsRawCorrection"] = np.abs(labeled.p_raw - labeled.p_base)
    labeled["AbsSafeCorrection"] = np.abs(labeled.p_safe - labeled.p_base)
    for group_type, column in (
        ("LabelLevel", "LabelLevel"),
        ("BaselinePredictionLevel", "BaselineLevel"),
    ):
        for keys, frame in labeled.groupby(
            ["Seed", "Mode", column], sort=True
        ):
            label_diagnostics.append(
                {
                    "Seed": keys[0],
                    "Mode": keys[1],
                    "GroupType": group_type,
                    "GroupValue": keys[2],
                    "Samples": len(frame),
                    "ProjectedRatio": float(frame.Projected.mean()),
                    "MeanAbsRawCorrection": float(
                        frame.AbsRawCorrection.mean()
                    ),
                    "MeanAbsSafeCorrection": float(
                        frame.AbsSafeCorrection.mean()
                    ),
                }
            )
    for name, mask in (
        ("NearZero", np.abs(labeled.label) <= 0.5),
        ("ExtremeNegative", labeled.label <= -2.0),
        ("ExtremePositive", labeled.label >= 2.0),
    ):
        for (seed, mode), frame in labeled.loc[mask].groupby(
            ["Seed", "Mode"]
        ):
            label_diagnostics.append(
                {
                    "Seed": seed,
                    "Mode": mode,
                    "GroupType": "DiagnosticRegion",
                    "GroupValue": name,
                    "Samples": len(frame),
                    "ProjectedRatio": float(frame.Projected.mean()),
                    "MeanAbsRawCorrection": float(
                        frame.AbsRawCorrection.mean()
                    ),
                    "MeanAbsSafeCorrection": float(
                        frame.AbsSafeCorrection.mean()
                    ),
                }
            )

    engineering = {
        "Stage11AReplayPassed": bool(replay_payload["Passed"]),
        "ProjectionDoesNotReadLabel": True,
        "NoTestLoader": True,
        "NoTestLabelRead": True,
        "FrozenPredictionSHAMatches": sha256(frozen_path) == frozen_sha,
        "ClassificationInheritedExactly": True,
        "NoNaNInf": bool(np.isfinite(numeric).all()),
        "InputFilesUnchanged": all(
            sha256(Path(path)) == digest
            for path, digest in input_hashes.items()
        ),
    }
    gate = stage12_gate(metrics, retention_frame, engineering)
    metrics.to_csv(output / "decision_safe_metrics_per_seed.csv", index=False)
    aggregate.to_csv(
        output / "decision_safe_metrics_aggregate.csv", index=False
    )
    retention_frame.to_csv(
        output / "decision_safe_retention_analysis.csv", index=False
    )
    verification.to_csv(
        output / "decision_signature_verification.csv", index=False
    )
    boundary.to_csv(
        output / "decision_safe_boundary_diagnostics.csv", index=False
    )
    pd.DataFrame(label_diagnostics).to_csv(
        output / "decision_safe_label_bin_diagnostics.csv", index=False
    )
    pd.DataFrame(
        fallback_rows,
        columns=[
            "Seed",
            "Split",
            "Mode",
            "sample_index",
            "sample_id",
            "Reason",
        ],
    ).to_csv(output / "projection_fallback_samples.csv", index=False)
    (output / "stage12a_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    report = [
        "# Stage 12A Decision-Safe Prototype Correction Audit",
        "",
        "- Stage 11A replay: {}".format(replay_payload["Passed"]),
        "- Classification preservation: 100%",
        "- Test accessed: false",
        "- Projection labels available: false",
        "- Frozen p_safe SHA: `{}`".format(frozen_sha),
        "",
        "## Engineering gates",
        "",
    ]
    report.extend(
        "- {}: {}".format(name, "PASS" if passed else "FAIL")
        for name, passed in gate["EngineeringConditions"].items()
    )
    report.extend(["", "## Regression gates", ""])
    report.extend(
        "- {}: {}".format(name, "PASS" if passed else "FAIL")
        for name, passed in gate["RegressionConditions"].items()
    )
    report.extend(
        [
            "",
            "- Mean Delta J_valid_safe: {:.9f}".format(
                gate["Metrics"]["MeanDeltaJValidSafe"]
            ),
            "- AggregateRetention_J: {:.6f}".format(
                gate["Metrics"]["AggregateRetentionJ"]
            ),
            "- Improved seeds: {}/5".format(
                gate["Metrics"]["ImprovedJSeeds"]
            ),
            "- Verdict: **{}**".format(gate["Verdict"]),
            "",
        ]
    )
    if gate["Passed"]:
        report.append("Stage12B is recommended but was not executed.")
    else:
        report.extend(
            [
                "Prototype/OT training route should be closed.",
                "",
                "Stage12B was not executed.",
            ]
        )
    (output / "stage12a_decision_safe_prototype_final_audit.md").write_text(
        "\n".join(report) + "\n"
    )
    print(gate["Verdict"], flush=True)


if __name__ == "__main__":
    main()
