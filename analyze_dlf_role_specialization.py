"""Frozen DLF role-specialization and long-tail audit.

This script never constructs the official test split and never updates a model.
It audits label imbalance and fits fixed linear probes on frozen train
representations, evaluating them once on official valid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataset
from train_cf_compat_kd import batch_to_device
from trains.singleTask.dlf_role_specialization_utils import (
    FORMAL_SEEDS,
    INTENSITY_NAMES,
    METHOD,
    MODES,
    NEUTRAL_TAU,
    OUTPUT_TAG,
    POLARITY_NAMES,
    REPRESENTATIONS,
    SENTIMENT_BINS,
    VERSION,
    FixedOrdinalProbe,
    FixedPolarityProbe,
    RepresentationCapture,
    bin_risk_metrics,
    effective_number_weights,
    intensity_labels,
    long_tail_status,
    normalize_sample_id,
    ordinal_probe_metrics,
    parse_video_id,
    polarity_labels,
    polarity_probe_metrics,
    role_alignment_gate,
    sentiment_bins,
    tensor_state_sha256,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


SOURCE_TAG = "cfcompat_sam_v1"
SOURCE_SUMMARY = "sam_valid_screen_summary.json"
SOURCE_AUDIT = "sam_valid_screen_audit_check.json"
SOURCE_GRID = "sam_valid_grid_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen DLF role-specialization and long-tail audit."
    )
    parser.add_argument("--probe-dataset", choices=("mosi",), default="mosi")
    parser.add_argument(
        "--distribution-datasets", nargs="+", default=["mosi", "mosei"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if tuple(args.seeds) != FORMAL_SEEDS:
        parser.error("Formal role audit fixes seeds to 1111 1114.")
    if tuple(args.distribution_datasets) != ("mosi", "mosei"):
        parser.error("Formal distribution audit fixes datasets to mosi mosei.")
    if args.num_workers != 1:
        parser.error("Formal role audit fixes num_workers=1.")
    return args


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Unsupported JSON value: {}".format(type(value)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_root(cli) -> Path:
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.probe_dataset
        / "valid_train_audit"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def source_root(cli) -> Path:
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / SOURCE_TAG
        / cli.probe_dataset
        / "valid_screen"
    )


def build_args(cli, dataset: str, seed: int):
    args = get_config_regression("DLF", dataset, cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(seed)
    args.device = assign_gpu(list(cli.gpu_ids))
    return args


def resolve_existing_path(value, cli) -> Path:
    path = Path(str(value))
    candidates = [
        path,
        Path.cwd() / path,
        Path(cli.result_root).resolve().parent / path,
        Path(cli.model_save_dir).resolve().parent / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Unable to resolve source path {}. Tried: {}".format(
            value, [str(candidate) for candidate in candidates]
        )
    )


def load_source(cli) -> Dict[str, object]:
    root = source_root(cli)
    required = {
        "summary": root / SOURCE_SUMMARY,
        "audit": root / SOURCE_AUDIT,
        "grid": root / SOURCE_GRID,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited SAM source artifacts:\n" + "\n".join(missing))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    audit = json.loads(required["audit"].read_text(encoding="utf-8"))
    grid = pd.read_csv(required["grid"])
    if not audit.get("passed", False):
        raise RuntimeError("The SAM source artifact did not pass its independent audit.")
    if summary["protocol"].get("official_test_constructed", True):
        raise RuntimeError("Source artifact unexpectedly constructed official Test.")
    baseline = grid.loc[
        grid.Seed.astype(int).isin(FORMAL_SEEDS)
        & grid.Run.astype(str).eq("adam_replay")
        & grid.Optimizer.astype(str).eq("Adam")
    ].copy()
    if sorted(baseline.Seed.astype(int).tolist()) != sorted(FORMAL_SEEDS):
        raise RuntimeError("Source grid lacks the two frozen Adam replay checkpoints.")
    records = []
    for row in baseline.to_dict("records"):
        checkpoint = resolve_existing_path(row["MainCheckpoint"], cli)
        if checkpoint_sha256(checkpoint) != str(row["MainCheckpointSHA256"]):
            raise RuntimeError("Source checkpoint hash mismatch: {}".format(checkpoint))
        row["ResolvedCheckpoint"] = str(checkpoint)
        records.append(row)
    return {
        "root": str(root.resolve()),
        "summary_path": str(required["summary"].resolve()),
        "summary_sha256": sha256_file(required["summary"]),
        "audit_path": str(required["audit"].resolve()),
        "audit_sha256": sha256_file(required["audit"]),
        "grid_path": str(required["grid"].resolve()),
        "grid_sha256": sha256_file(required["grid"]),
        "baseline_rows": records,
    }


def feature_file(args) -> Path:
    path = Path(str(args.featurePath))
    if path.is_file():
        return path.resolve()
    rooted = Path.cwd() / path
    return rooted.resolve()


def dataset_sample_frame(dataset_name: str, split: str, dataset: MMDataset) -> pd.DataFrame:
    labels = np.asarray(dataset.labels["M"], dtype=np.float64).reshape(-1)
    ids = [normalize_sample_id(value) for value in list(dataset.ids)]
    if len(labels) != len(ids):
        raise RuntimeError("Dataset labels and IDs differ in length.")
    return pd.DataFrame({
        "Dataset": str(dataset_name),
        "Split": str(split),
        "sample_index": np.arange(len(labels), dtype=np.int64),
        "sample_id": ids,
        "video_id": [parse_video_id(value) for value in ids],
        "label": labels,
        "polarity": polarity_labels(labels),
        "intensity": intensity_labels(labels),
        "sentiment_bin": sentiment_bins(labels),
    })


def distribution_rows(frame: pd.DataFrame) -> List[Dict[str, object]]:
    families = {
        "polarity_3": ("polarity", range(3)),
        "intensity_4": ("intensity", range(4)),
        "sentiment_7": ("sentiment_bin", SENTIMENT_BINS),
    }
    rows: List[Dict[str, object]] = []
    for family, (column, classes) in families.items():
        for class_value in classes:
            local = frame.loc[frame[column].astype(int).eq(int(class_value))]
            rows.append({
                "Dataset": str(frame.Dataset.iloc[0]),
                "Split": str(frame.Split.iloc[0]),
                "LabelFamily": family,
                "Class": int(class_value),
                "Count": int(len(local)),
                "Fraction": float(len(local) / len(frame)),
                "VideoCount": int(local.video_id.nunique()),
                "MeanLabel": float(local.label.mean()) if len(local) else float("nan"),
                "MinLabel": float(local.label.min()) if len(local) else float("nan"),
                "MaxLabel": float(local.label.max()) if len(local) else float("nan"),
            })
    return rows


def audit_distributions(cli, root: Path):
    availability = []
    sample_frames = []
    distribution = []
    weight_frames = []
    for dataset_name in cli.distribution_datasets:
        args = build_args(cli, dataset_name, seed=FORMAL_SEEDS[0])
        path = feature_file(args)
        if not path.is_file():
            if dataset_name == cli.probe_dataset:
                raise FileNotFoundError("Required MOSI feature file is absent: {}".format(path))
            availability.append({
                "Dataset": dataset_name,
                "Available": False,
                "FeaturePath": str(path),
                "Reason": "feature_file_absent",
            })
            continue
        availability.append({
            "Dataset": dataset_name,
            "Available": True,
            "FeaturePath": str(path),
            "FeatureSHA256": sha256_file(path),
            "Reason": "available",
        })
        for split in ("train", "valid"):
            dataset = MMDataset(args, mode=split)
            frame = dataset_sample_frame(dataset_name, split, dataset)
            sample_frames.append(frame)
            distribution.extend(distribution_rows(frame))
            if split == "train":
                for family, column, classes in (
                    ("polarity_3", "polarity", range(3)),
                    ("intensity_4", "intensity", range(4)),
                    ("sentiment_7", "sentiment_bin", SENTIMENT_BINS),
                ):
                    counts = {
                        int(class_value): int(
                            frame[column].astype(int).eq(int(class_value)).sum()
                        )
                        for class_value in classes
                    }
                    local_weights = effective_number_weights(counts)
                    local_weights.insert(0, "LabelFamily", family)
                    local_weights.insert(0, "Dataset", dataset_name)
                    weight_frames.append(local_weights)
    samples = pd.concat(sample_frames, ignore_index=True)
    distribution_frame = pd.DataFrame(distribution)
    weights = pd.concat(weight_frames, ignore_index=True)
    availability_frame = pd.DataFrame(availability)
    samples.to_csv(root / "label_samples_train_valid.csv", index=False)
    distribution_frame.to_csv(root / "label_distribution.csv", index=False)
    weights.to_csv(root / "effective_number_weights.csv", index=False)
    availability_frame.to_csv(root / "dataset_availability.csv", index=False)
    tail = {
        dataset: long_tail_status(distribution_frame, dataset)
        for dataset in availability_frame.loc[
            availability_frame.Available.astype(bool)
        ].Dataset.astype(str)
    }
    return samples, distribution_frame, weights, availability_frame, tail


def load_student(args, checkpoint: Path) -> MissingModalityWrapper:
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if model.training:
        raise RuntimeError("Frozen representation model must be in eval mode.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def aligned_batch_ids(value, batch_size: int) -> List[str]:
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size:
            return [normalize_sample_id(item) for item in value]
        components = []
        for item in value:
            if torch.is_tensor(item) and item.numel() == batch_size:
                components.append(item.detach().cpu().tolist())
            elif isinstance(item, np.ndarray) and len(item) == batch_size:
                components.append(item.tolist())
            elif isinstance(item, (list, tuple)) and len(item) == batch_size:
                components.append(list(item))
        if components:
            return [
                normalize_sample_id(tuple(component[index] for component in components))
                for index in range(batch_size)
            ]
    if torch.is_tensor(value) and value.size(0) == batch_size:
        return [normalize_sample_id(item) for item in value]
    raise RuntimeError("Unable to align batch IDs for batch size {}.".format(batch_size))


def representation_file(root: Path, seed: int, split: str, mode: str) -> Path:
    directory = root / "representations" / "seed{}".format(int(seed))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "{}_{}.npz".format(split, mode)


def extract_representations(cli, root: Path, source: Mapping[str, object]):
    prediction_rows = []
    manifest_rows = []
    state_rows = []
    checkpoint_by_seed = {
        int(row["Seed"]): Path(str(row["ResolvedCheckpoint"]))
        for row in source["baseline_rows"]
    }
    for seed in FORMAL_SEEDS:
        setup_seed(seed)
        args = build_args(cli, cli.probe_dataset, seed)
        checkpoint = checkpoint_by_seed[seed]
        model = load_student(args, checkpoint)
        state_before = tensor_state_sha256(model)
        with RepresentationCapture(model) as capture:
            for split in ("train", "valid"):
                loader = build_single_split_loader(args, split, cli.num_workers)
                for mode in MODES:
                    arrays: Dict[str, List[np.ndarray]] = {
                        "shared": [],
                        "specific_l": [],
                        "specific_a": [],
                        "specific_v": [],
                        "specific_present": [],
                        "final_fusion": [],
                    }
                    labels_all = []
                    indices_all = []
                    ids_all: List[str] = []
                    predictions_all = []
                    with torch.no_grad():
                        for batch in loader:
                            text, audio, vision, labels = batch_to_device(batch, args.device)
                            mask = mode_to_mask(
                                mode,
                                batch_size=labels.size(0),
                                device=args.device,
                                dtype=audio.dtype,
                            )
                            output = model(text, audio, vision, mask)
                            captured = capture.take(mode)
                            for name in arrays:
                                arrays[name].append(
                                    captured[name].numpy().astype(np.float32, copy=False)
                                )
                            batch_indices = (
                                batch["index"].view(-1).cpu().numpy().astype(np.int64)
                            )
                            batch_ids = aligned_batch_ids(batch["id"], labels.size(0))
                            if len(batch_ids) != labels.size(0):
                                raise RuntimeError("Batch ID count mismatch.")
                            labels_np = labels.detach().cpu().view(-1).numpy().astype(np.float64)
                            predictions_np = (
                                output["output_logit"].detach().cpu().view(-1).numpy().astype(np.float64)
                            )
                            labels_all.append(labels_np)
                            indices_all.append(batch_indices)
                            ids_all.extend(batch_ids)
                            predictions_all.append(predictions_np)
                    payload = {
                        name: np.concatenate(values, axis=0)
                        for name, values in arrays.items()
                    }
                    payload["label"] = np.concatenate(labels_all, axis=0)
                    payload["sample_index"] = np.concatenate(indices_all, axis=0)
                    payload["sample_id"] = np.asarray(ids_all, dtype=np.str_)
                    payload["prediction"] = np.concatenate(predictions_all, axis=0)
                    payload["polarity"] = polarity_labels(payload["label"])
                    payload["intensity"] = intensity_labels(payload["label"])
                    payload["sentiment_bin"] = sentiment_bins(payload["label"])
                    path = representation_file(root, seed, split, mode)
                    np.savez_compressed(path, **payload)
                    manifest_rows.append({
                        "Seed": int(seed),
                        "Split": split,
                        "Mode": mode,
                        "Path": str(path.resolve()),
                        "SHA256": sha256_file(path),
                        "SampleCount": int(len(payload["label"])),
                        "SharedDim": int(payload["shared"].shape[1]),
                        "SpecificPresentDim": int(payload["specific_present"].shape[1]),
                        "FinalFusionDim": int(payload["final_fusion"].shape[1]),
                    })
                    for index in range(len(payload["label"])):
                        prediction_rows.append({
                            "Seed": int(seed),
                            "Split": split,
                            "Mode": mode,
                            "sample_index": int(payload["sample_index"][index]),
                            "sample_id": str(payload["sample_id"][index]),
                            "video_id": parse_video_id(payload["sample_id"][index]),
                            "label": float(payload["label"][index]),
                            "prediction": float(payload["prediction"][index]),
                            "polarity": int(payload["polarity"][index]),
                            "intensity": int(payload["intensity"][index]),
                            "sentiment_bin": int(payload["sentiment_bin"][index]),
                        })
        state_after = tensor_state_sha256(model)
        state_rows.append({
            "Seed": int(seed),
            "Checkpoint": str(checkpoint.resolve()),
            "CheckpointSHA256": checkpoint_sha256(checkpoint),
            "StateSHA256Before": state_before,
            "StateSHA256After": state_after,
            "ParametersUnchanged": bool(state_before == state_after),
            "TrainableParameterCountDuringAudit": int(
                sum(parameter.requires_grad for parameter in model.parameters())
            ),
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predictions = pd.DataFrame(prediction_rows)
    manifest = pd.DataFrame(manifest_rows)
    states = pd.DataFrame(state_rows)
    predictions.to_csv(root / "baseline_predictions_train_valid.csv", index=False)
    manifest.to_csv(root / "representation_manifest.csv", index=False)
    states.to_csv(root / "model_state_audit.csv", index=False)
    return predictions, manifest, states


def load_representation(root: Path, seed: int, split: str, mode: str):
    path = representation_file(root, seed, split, mode)
    if not path.is_file():
        raise FileNotFoundError("Representation file absent: {}".format(path))
    return np.load(path, allow_pickle=False)


def available_representation_names(mode: str) -> Tuple[str, ...]:
    names = list(REPRESENTATIONS)
    names.append("specific_l")
    if "A" in mode:
        names.append("specific_a")
    if "V" in mode:
        names.append("specific_v")
    return tuple(names)


def run_probes(root: Path):
    metric_rows = []
    prediction_rows = []
    for seed in FORMAL_SEEDS:
        for mode in MODES:
            train = load_representation(root, seed, "train", mode)
            valid = load_representation(root, seed, "valid", mode)
            train_polarity = train["polarity"].astype(np.int64)
            valid_polarity = valid["polarity"].astype(np.int64)
            train_intensity = train["intensity"].astype(np.int64)
            valid_intensity = valid["intensity"].astype(np.int64)
            for representation in available_representation_names(mode):
                train_x = train[representation].astype(np.float64)
                valid_x = valid[representation].astype(np.float64)
                polarity_probe = FixedPolarityProbe.fit(train_x, train_polarity)
                polarity_prediction = polarity_probe.predict(valid_x)
                polarity_metrics = polarity_probe_metrics(
                    valid_polarity, polarity_prediction
                )
                metric_rows.append({
                    "Seed": int(seed),
                    "Mode": mode,
                    "Representation": representation,
                    "Task": "polarity_3",
                    "TrainCount": int(len(train_x)),
                    "ValidCount": int(len(valid_x)),
                    "FeatureDim": int(train_x.shape[1]),
                    **polarity_metrics,
                    "ordinal_mae": float("nan"),
                    "quadratic_kappa": float("nan"),
                })
                ordinal_probe = FixedOrdinalProbe.fit(train_x, train_intensity)
                ordinal_prediction, ordinal_probabilities = ordinal_probe.predict(valid_x)
                ordinal_metrics = ordinal_probe_metrics(
                    valid_intensity, ordinal_prediction
                )
                metric_rows.append({
                    "Seed": int(seed),
                    "Mode": mode,
                    "Representation": representation,
                    "Task": "absolute_intensity_ordinal_4",
                    "TrainCount": int(len(train_x)),
                    "ValidCount": int(len(valid_x)),
                    "FeatureDim": int(train_x.shape[1]),
                    "balanced_accuracy": float("nan"),
                    **ordinal_metrics,
                })
                for index in range(len(valid_x)):
                    prediction_rows.append({
                        "Seed": int(seed),
                        "Mode": mode,
                        "Representation": representation,
                        "sample_index": int(valid["sample_index"][index]),
                        "sample_id": str(valid["sample_id"][index]),
                        "label": float(valid["label"][index]),
                        "true_polarity": int(valid_polarity[index]),
                        "pred_polarity": int(polarity_prediction[index]),
                        "true_intensity": int(valid_intensity[index]),
                        "pred_intensity": int(ordinal_prediction[index]),
                        "p_gt_0p5": float(ordinal_probabilities[index, 0]),
                        "p_gt_1p5": float(ordinal_probabilities[index, 1]),
                        "p_gt_2p5": float(ordinal_probabilities[index, 2]),
                    })
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(root / "probe_metrics.csv", index=False)
    predictions.to_csv(root / "probe_valid_predictions.csv", index=False)
    comparison_rows = []
    for seed in FORMAL_SEEDS:
        for mode in MODES:
            polarity = metrics.loc[
                metrics.Seed.astype(int).eq(seed)
                & metrics.Mode.astype(str).eq(mode)
                & metrics.Task.astype(str).eq("polarity_3")
            ]
            intensity = metrics.loc[
                metrics.Seed.astype(int).eq(seed)
                & metrics.Mode.astype(str).eq(mode)
                & metrics.Task.astype(str).eq("absolute_intensity_ordinal_4")
            ]
            shared_pol = polarity.loc[
                polarity.Representation.astype(str).eq("shared")
            ].iloc[0]
            specific_pol = polarity.loc[
                polarity.Representation.astype(str).eq("specific_present")
            ].iloc[0]
            shared_int = intensity.loc[
                intensity.Representation.astype(str).eq("shared")
            ].iloc[0]
            specific_int = intensity.loc[
                intensity.Representation.astype(str).eq("specific_present")
            ].iloc[0]
            comparison_rows.append({
                "Seed": int(seed),
                "Mode": mode,
                "shared_polarity_macro_f1": float(shared_pol.macro_f1),
                "specific_polarity_macro_f1": float(specific_pol.macro_f1),
                "polarity_advantage": float(
                    shared_pol.macro_f1 - specific_pol.macro_f1
                ),
                "shared_intensity_ordinal_mae": float(shared_int.ordinal_mae),
                "specific_intensity_ordinal_mae": float(specific_int.ordinal_mae),
                "intensity_advantage": float(
                    shared_int.ordinal_mae - specific_int.ordinal_mae
                ),
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(root / "role_alignment_comparisons.csv", index=False)
    gate = role_alignment_gate(comparisons)
    return metrics, predictions, comparisons, gate


def baseline_risk_audit(predictions: pd.DataFrame, root: Path):
    bin_frames = []
    summary_rows = []
    for (seed, split, mode), local in predictions.groupby(
        ["Seed", "Split", "Mode"], sort=True
    ):
        bins, summary = bin_risk_metrics(local)
        bins.insert(0, "Mode", str(mode))
        bins.insert(0, "Split", str(split))
        bins.insert(0, "Seed", int(seed))
        bin_frames.append(bins)
        summary_rows.append({
            "Seed": int(seed),
            "Split": str(split),
            "Mode": str(mode),
            **summary,
        })
    bins = pd.concat(bin_frames, ignore_index=True)
    summaries = pd.DataFrame(summary_rows)
    bins.to_csv(root / "baseline_bin_risk.csv", index=False)
    summaries.to_csv(root / "baseline_risk_summary.csv", index=False)
    return bins, summaries


def render_report(summary: Mapping[str, object]) -> str:
    role = summary["role_alignment"]
    mosi_tail = summary["long_tail"].get("mosi", {})
    lines = [
        "# DLF role-specialization and long-tail audit v1",
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Role status: `{}`".format(role["status"]),
        "- Role gate passed: `{}`".format(role["passed"]),
        "- MOSI long tail present: `{}`".format(mosi_tail.get("tail_present")),
        "- MOSI seven-bin imbalance ratio: `{:.4f}`".format(
            float(mosi_tail.get("imbalance_ratio_max_over_min", float("nan")))
        ),
        "- Official Test constructed: `False`",
        "- Model parameters updated: `False`",
        "",
        "## Role-alignment evidence",
        "",
        "- Mean shared polarity advantage: `{:+.6f}`".format(
            role["mean_polarity_advantage"]
        ),
        "- Mean specific intensity advantage: `{:+.6f}`".format(
            role["mean_intensity_advantage"]
        ),
        "- Polarity positive coverage: `{:.3f}`".format(
            role["polarity_positive_coverage"]
        ),
        "- Intensity positive coverage: `{:.3f}`".format(
            role["intensity_positive_coverage"]
        ),
        "",
        "## Interpretation",
        "",
    ]
    if summary["verdict"] == "PROMOTE_ROLE_SPECIALIZATION_STAGE_B":
        lines.append(
            "The frozen representations support the proposed semantic role split. "
            "The next experiment may test semantic specialization only, without "
            "tail weights or semantic-risk loss."
        )
    elif summary["verdict"] == "PARTIAL_ROLE_ALIGNMENT_DO_NOT_TRAIN_YET":
        lines.append(
            "The average direction is favorable but the preregistered stability "
            "gate is incomplete. Review per-seed and per-view failures before any "
            "training change."
        )
    elif summary["verdict"] == "STOP_LONG_TAIL_PREMISE_NOT_SUPPORTED":
        lines.append(
            "The MOSI train distribution did not meet the frozen long-tail criterion."
        )
    else:
        lines.append(
            "The frozen DLF spaces do not support assigning polarity and absolute "
            "intensity to shared and specific representations respectively."
        )
    return "\n".join(lines) + "\n"


def main():
    cli = parse_args()
    root = output_root(cli)
    source = load_source(cli)
    samples, distribution, weights, availability, tail = audit_distributions(
        cli, root
    )
    predictions, representation_manifest, states = extract_representations(
        cli, root, source
    )
    probe_metrics, probe_predictions, comparisons, role_gate = run_probes(root)
    risk_bins, risk_summary = baseline_risk_audit(predictions, root)

    if not tail["mosi"]["tail_present"]:
        verdict = "STOP_LONG_TAIL_PREMISE_NOT_SUPPORTED"
    elif role_gate["passed"]:
        verdict = "PROMOTE_ROLE_SPECIALIZATION_STAGE_B"
    elif role_gate["status"] == "PARTIAL_ROLE_ALIGNMENT_NEEDS_REVIEW":
        verdict = "PARTIAL_ROLE_ALIGNMENT_DO_NOT_TRAIN_YET"
    else:
        verdict = "STOP_ROLE_SPECIALIZATION_HYPOTHESIS"

    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": "feature/cfcompat-sam-valid-screen-v1",
        "source": source,
        "formal_seeds": list(FORMAL_SEEDS),
        "probe_dataset": cli.probe_dataset,
        "distribution_datasets": list(cli.distribution_datasets),
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "optimizer_constructed": False,
        "backward_called": False,
        "model_parameters_updated": False,
        "neutral_tau": NEUTRAL_TAU,
        "probe_hyperparameters": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 0,
            "selection_on_valid": False,
        },
        "artifacts": {},
    }
    artifact_names = [
        "label_samples_train_valid.csv",
        "label_distribution.csv",
        "effective_number_weights.csv",
        "dataset_availability.csv",
        "baseline_predictions_train_valid.csv",
        "representation_manifest.csv",
        "model_state_audit.csv",
        "probe_metrics.csv",
        "probe_valid_predictions.csv",
        "role_alignment_comparisons.csv",
        "baseline_bin_risk.csv",
        "baseline_risk_summary.csv",
    ]
    for name in artifact_names:
        path = root / name
        manifest["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    (root / "role_specialization_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "long_tail": tail,
        "role_alignment": role_gate,
        "protocol": {
            "fit_split": "official_train_only",
            "evaluation_split": "official_valid_only",
            "official_test_constructed": False,
            "official_test_authorized": False,
            "model_parameters_updated": False,
            "representation_families": list(REPRESENTATIONS),
            "modes": list(MODES),
            "neutral_tau": NEUTRAL_TAU,
            "specific_representation_uses_present_modalities_only": True,
            "next_stage_on_pass": "semantic_role_specialization_without_tail_or_risk",
        },
        "counts": {
            "label_samples": int(len(samples)),
            "distribution_rows": int(len(distribution)),
            "weight_rows": int(len(weights)),
            "baseline_prediction_rows": int(len(predictions)),
            "representation_files": int(len(representation_manifest)),
            "probe_metric_rows": int(len(probe_metrics)),
            "probe_prediction_rows": int(len(probe_predictions)),
            "role_comparison_rows": int(len(comparisons)),
            "risk_bin_rows": int(len(risk_bins)),
            "risk_summary_rows": int(len(risk_summary)),
        },
        "model_state_all_unchanged": bool(states.ParametersUnchanged.astype(bool).all()),
        "mosei_distribution_available": bool(
            availability.loc[
                availability.Dataset.astype(str).eq("mosei"), "Available"
            ].astype(bool).any()
        ),
    }
    (root / "role_specialization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    (root / "role_specialization_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print("DLF role-specialization audit complete")
    print("verdict:", verdict)
    print("role status:", role_gate["status"])
    print(
        "mean shared polarity advantage:",
        "{:+.6f}".format(role_gate["mean_polarity_advantage"]),
    )
    print(
        "mean specific intensity advantage:",
        "{:+.6f}".format(role_gate["mean_intensity_advantage"]),
    )
    print("official Test was not constructed")
    print("report:", root / "role_specialization_report.md")


if __name__ == "__main__":
    main()
