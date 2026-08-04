"""Frozen MOSI dataset limitations and CFCompatKD mechanism audit.

The script constructs train/valid datasets and validation-best models only.
Official Test is never constructed, read, or traversed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_loader import MMDataset
from train_cf_compat_kd import build_config
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    build_frozen_evaluator,
    load_counterfactual_cache,
)
from trains.singleTask.cfcompat_stability_utils import load_stage3_reference
from trains.singleTask.fixed_kd_utils import (
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MODALITY_MASKS,
    MissingModalityWrapper,
    mode_to_mask,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from trains.singleTask.mosi_cfcompat_audit_utils import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FORMAL_SEEDS,
    METHOD,
    MODES,
    OUTPUT_TAG,
    VERSION,
    dataset_limitation_flags,
    dataset_sample_frame,
    dataset_split_summary,
    empirical_compatibility,
    feature_quality_rows,
    group_mechanism_summary,
    joint_video_bootstrap,
    mechanism_assessment,
    modality_marginal_value,
    opportunity_ranking,
    overall_prediction_summary,
    prediction_events,
    sha256_file,
    split_shift_summary,
    video_summary,
)
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit MOSI limitations and the frozen CFCompatKD gain mechanism."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if tuple(args.seeds) != FORMAL_SEEDS:
        parser.error("The formal audit fixes seeds to 1111 1114.")
    if int(args.num_workers) != 1:
        parser.error("The formal audit fixes num_workers=1.")
    if int(args.bootstrap_replicates) != BOOTSTRAP_REPLICATES:
        parser.error("The formal audit fixes 2000 joint video bootstrap replicates.")
    return args


def output_root(cli) -> Path:
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / "mosi"
        / "train_valid_audit"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_checkpoint_from_row(row: pd.Series) -> Path:
    for field in (
        "MainCheckpoint",
        "Checkpoint",
        "BestCheckpoint",
        "ModelCheckpoint",
    ):
        if field in row.index and pd.notna(row[field]):
            checkpoint = Path(str(row[field]))
            if checkpoint.is_file():
                return checkpoint
            if not checkpoint.is_absolute():
                alternatives = [
                    Path.cwd() / checkpoint,
                    Path("/code/DLF") / checkpoint,
                ]
                for alternative in alternatives:
                    if alternative.is_file():
                        return alternative
    raise FileNotFoundError("No recorded checkpoint path in source row is available.")


def load_unique_seed_row(path: Path, seed: int) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "Seed" not in frame.columns:
        raise ValueError("Source result has no Seed column: {}".format(path))
    selected = frame.loc[frame.Seed.astype(int).eq(int(seed))]
    if len(selected) != 1:
        raise RuntimeError("Source result is not unique for seed {}: {}".format(seed, path))
    return selected.iloc[0]


def baseline_source(result_root: Path, seed: int):
    if int(seed) == 1111:
        path = result_root / "missing_baseline/moddrop/train/mosi_per_seed.csv"
    else:
        path = (
            result_root
            / "missing_baseline/moddrop_benchmark_multiseed_v1"
            / "seed{}".format(seed)
            / "mosi_per_seed.csv"
        )
    row = load_unique_seed_row(path, seed)
    checkpoint = resolve_checkpoint_from_row(row)
    return row, path, checkpoint


def cfcompat_source(result_root: Path, seed: int):
    row, path = load_stage3_reference(result_root, seed)
    checkpoint = resolve_checkpoint_from_row(row)
    return row, path, checkpoint


def clean_teacher_checkpoint(model_save_dir: Path, seed: int) -> Path:
    path = model_save_dir / "DLF_mosi_seed{}_best.pth".format(seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evaluator_and_cache(cli, args, seed: int, baseline_checkpoint: Path):
    evaluator_sha = checkpoint_sha256(baseline_checkpoint)
    version = CACHE_VERSION if int(seed) == 1111 else MULTISEED_CACHE_VERSION
    cache_seed = None if int(seed) == 1111 else int(seed)
    cache, by_index = load_counterfactual_cache(
        cli.result_root,
        "mosi",
        version=version,
        seed=cache_seed,
        expected_evaluator_sha=evaluator_sha,
    )
    if len(cache) != 1284:
        raise RuntimeError("MOSI compatibility cache must contain 1284 train samples.")
    evaluator = build_frozen_evaluator(DLF, args, baseline_checkpoint)
    return evaluator, cache, by_index


def load_cfcompat_model(args, checkpoint: Path):
    model = build_frozen_evaluator(DLF, args, checkpoint)
    if not isinstance(model, MissingModalityWrapper):
        raise TypeError("CFCompat checkpoint did not load as MissingModalityWrapper.")
    return model


def model_predictions(model, text, audio, vision, mode):
    mask = mode_to_mask(mode, text.size(0), audio.device, audio.dtype)
    with torch.inference_mode():
        prediction = model(text, audio, vision, mask)["output_logit"].detach().view(-1)
    return prediction.cpu().numpy().astype(np.float64)


def evaluate_seed(cli, seed: int, valid_dataset, metadata: pd.DataFrame):
    setup_seed(seed)
    args = build_config(cli, seed)
    args.mode = "train"
    baseline_row, baseline_csv, baseline_checkpoint = baseline_source(
        Path(cli.result_root), seed
    )
    cf_row, cf_csv, cf_checkpoint = cfcompat_source(Path(cli.result_root), seed)
    teacher_checkpoint = clean_teacher_checkpoint(Path(cli.model_save_dir), seed)

    evaluator, train_cache, _ = evaluator_and_cache(
        cli, args, seed, baseline_checkpoint
    )
    cfcompat = load_cfcompat_model(args, cf_checkpoint)
    teacher = build_frozen_teacher(DLF, args, teacher_checkpoint)

    batch_size = int(args.batch_size)
    rows = []
    for start in range(0, len(valid_dataset), batch_size):
        stop = min(len(valid_dataset), start + batch_size)
        indices = np.arange(start, stop, dtype=int)
        text = torch.as_tensor(valid_dataset.text[indices], device=args.device)
        audio = torch.as_tensor(valid_dataset.audio[indices], device=args.device)
        vision = torch.as_tensor(valid_dataset.vision[indices], device=args.device)
        labels = np.asarray(valid_dataset.labels["M"][indices], dtype=np.float64).reshape(-1)
        with torch.inference_mode():
            teacher_prediction = (
                teacher_lav_prediction(teacher, text, audio, vision)
                .view(-1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        evaluator_predictions = {
            mode: model_predictions(evaluator, text, audio, vision, mode)
            for mode in MODES
        }
        cf_predictions = {
            mode: model_predictions(cfcompat, text, audio, vision, mode)
            for mode in MODES
        }
        for offset, index in enumerate(indices):
            base_meta = metadata.loc[metadata.sample_index.eq(int(index))]
            if len(base_meta) != 1:
                raise RuntimeError("Valid metadata binding is not unique.")
            base_meta = base_meta.iloc[0]
            for mode in MODES:
                shift = abs(
                    float(evaluator_predictions["LAV"][offset])
                    - float(evaluator_predictions[mode][offset])
                )
                if mode == "LAV":
                    compatibility = 1.0
                else:
                    compatibility = float(
                        empirical_compatibility(
                            train_cache["delta_{}".format(mode)].to_numpy(dtype=float),
                            [shift],
                        )[0]
                    )
                rows.append(
                    {
                        "Seed": int(seed),
                        "Mode": mode,
                        "sample_index": int(index),
                        "sample_id": str(base_meta.sample_id),
                        "video_id": str(base_meta.video_id),
                        "segment_id": str(base_meta.segment_id),
                        "label": float(labels[offset]),
                        "baseline_prediction": float(evaluator_predictions[mode][offset]),
                        "cfcompat_prediction": float(cf_predictions[mode][offset]),
                        "teacher_prediction": float(teacher_prediction[offset]),
                        "evaluator_LAV_prediction": float(evaluator_predictions["LAV"][offset]),
                        "evaluator_shift": float(shift),
                        "compatibility_proxy": float(compatibility),
                        "token_count": int(base_meta.token_count),
                    }
                )

    predictions = prediction_events(pd.DataFrame(rows))
    sources = {
        "seed": int(seed),
        "baseline_result": str(baseline_csv.resolve()),
        "baseline_result_sha256": sha256_file(baseline_csv),
        "baseline_checkpoint": str(baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": checkpoint_sha256(baseline_checkpoint),
        "cfcompat_result": str(cf_csv.resolve()),
        "cfcompat_result_sha256": sha256_file(cf_csv),
        "cfcompat_checkpoint": str(cf_checkpoint.resolve()),
        "cfcompat_checkpoint_sha256": checkpoint_sha256(cf_checkpoint),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": checkpoint_sha256(teacher_checkpoint),
        "compatibility_cache_version": CACHE_VERSION if seed == 1111 else MULTISEED_CACHE_VERSION,
        "compatibility_cache_sample_count": int(len(train_cache)),
    }

    del evaluator, cfcompat, teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, sources, baseline_row, cf_row


def feature_audit(train_dataset, valid_dataset):
    rows = []
    for split, dataset in (("train", train_dataset), ("valid", valid_dataset)):
        rows.append(feature_quality_rows(split, "text_tensor", dataset.text))
        rows.append(feature_quality_rows(split, "audio", dataset.audio))
        rows.append(feature_quality_rows(split, "vision", dataset.vision))
    sample_quality = pd.concat(rows, ignore_index=True)
    summary = (
        sample_quality.groupby(["Split", "Modality"], as_index=False)
        .agg(
            sample_count=("sample_index", "size"),
            finite_fraction_mean=("finite_fraction", "mean"),
            effective_length_mean=("effective_length", "mean"),
            effective_length_std=("effective_length", "std"),
            zero_fraction_mean=("zero_fraction", "mean"),
            mean_abs_value=("mean_abs_value", "mean"),
            sample_std_mean=("sample_std", "mean"),
            all_zero_fraction=("all_zero", "mean"),
            near_constant_fraction=("near_constant", "mean"),
        )
    )
    return sample_quality, summary


def historical_control_summary(result_root: Path):
    paths = {
        1114: result_root / "missing_baseline/cfcompat_evidence_v1/mosi/stage18c_seed1114_controls/stage18c_seed1114_metrics.csv",
        1111: result_root / "missing_baseline/cfcompat_evidence_v1/mosi/stage18d_seed1111_replication/stage18d_seed1111_metrics.csv",
    }
    frames = []
    bindings = []
    for seed, path in paths.items():
        if not path.is_file():
            continue
        local = pd.read_csv(path)
        local["SourceSeed"] = int(seed)
        frames.append(local)
        bindings.append({"seed": int(seed), "path": str(path.resolve()), "sha256": sha256_file(path)})
    if not frames:
        return pd.DataFrame(), bindings
    combined = pd.concat(frames, ignore_index=True)
    wanted = [
        column
        for column in (
            "Seed", "SourceSeed", "Method", "J_valid",
            "valid_LAV_MAE", "valid_LA_MAE", "valid_LV_MAE", "valid_L_MAE",
        )
        if column in combined.columns
    ]
    return combined[wanted].copy(), bindings


def render_report(summary: dict, split_summary: pd.DataFrame, overall: pd.DataFrame, opportunities: pd.DataFrame, controls: pd.DataFrame) -> str:
    limitation = summary["dataset_limitations"]
    mechanism = summary["mechanism_assessment"]
    lines = [
        "# MOSI dataset limitations and CFCompatKD mechanism audit v1",
        "",
        "## Frozen decision",
        "",
        "- Verdict: `{}`".format(mechanism["verdict"]),
        "- Mean two-seed Valid J gain: `{:+.6f}`".format(mechanism["mean_two_seed_J_gain"]),
        "- Video-bootstrap 95% CI: `[{:+.6f}, {:+.6f}]`".format(
            mechanism["video_bootstrap_J_gain"]["ci95_low"],
            mechanism["video_bootstrap_J_gain"]["ci95_high"],
        ),
        "- Official Test constructed: `False`",
        "- Model training performed: `False`",
        "",
        "## Dataset support",
        "",
        "```",
        split_summary.to_string(index=False),
        "```",
        "",
        "## Limitation indicators",
        "",
    ]
    for key, value in limitation.items():
        lines.append("- {}: `{}`".format(key, value))
    lines.extend(
        [
            "",
            "## DLF-ModDrop versus CFCompatKD",
            "",
            "```",
            overall.to_string(index=False),
            "```",
            "",
            "## Mechanism checks",
            "",
        ]
    )
    for key, value in mechanism["checks"].items():
        lines.append("- {}: `{}`".format(key, value))
    lines.extend(
        [
            "",
            "The primary hypothesis is that CFCompatKD reduces indiscriminate "
            "imitation: it transfers the clean LAV Teacher only where a frozen "
            "missing-view evaluator changes little under the corresponding "
            "counterfactual removal. Compatibility is therefore an applicability "
            "signal, not a generic confidence or correctness estimate.",
            "",
            "## Descriptive opportunity ranking",
            "",
        ]
    )
    if opportunities.empty:
        lines.append("No group met the fixed two-seed minimum-support rule.")
    else:
        lines.extend(["```", opportunities.head(30).to_string(index=False), "```"])
    lines.extend(["", "## Historical frozen KD controls", ""])
    if controls.empty:
        lines.append("Stage 18 control metrics were unavailable and were not inferred.")
    else:
        lines.extend(["```", controls.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "This audit is descriptive. It does not authorize a new loss, router, "
            "tail weighting, Teacher subset, Test evaluation, or hyperparameter search.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    cli = parse_args()
    root = output_root(cli)
    setup_seed(1111)
    args = build_config(cli, 1111)
    args.mode = "train"
    train_dataset = MMDataset(args, mode="train")
    valid_dataset = MMDataset(args, mode="valid")

    train_samples = dataset_sample_frame(train_dataset, "train")
    valid_samples = dataset_sample_frame(valid_dataset, "valid")
    samples = pd.concat([train_samples, valid_samples], ignore_index=True)
    split_summary = dataset_split_summary(samples)
    videos = video_summary(samples)
    shift = split_shift_summary(samples)
    feature_samples, feature_summary = feature_audit(train_dataset, valid_dataset)

    prediction_frames = []
    source_records = []
    source_metric_rows = []
    for seed in FORMAL_SEEDS:
        predictions, sources, baseline_row, cf_row = evaluate_seed(
            cli, seed, valid_dataset, valid_samples
        )
        prediction_frames.append(predictions)
        source_records.append(sources)
        source_metric_rows.append(
            {
                "Seed": int(seed),
                "baseline_source_J": float(baseline_row.get("J_valid", baseline_row.get("J_val", np.nan))),
                "cfcompat_source_J": float(cf_row.get("J_valid", np.nan)),
                "baseline_source_epoch": int(baseline_row.get("BestValidEpoch", baseline_row.get("BestEpoch", -1))),
                "cfcompat_source_epoch": int(cf_row.get("BestValidEpoch", -1)),
            }
        )
    events = pd.concat(prediction_frames, ignore_index=True)
    overall = overall_prediction_summary(events)
    groups = group_mechanism_summary(events)
    modality_value = modality_marginal_value(events)
    bootstrap = joint_video_bootstrap(
        events,
        replicates=cli.bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    opportunities = opportunity_ranking(groups)
    limitations = dataset_limitation_flags(split_summary, shift, modality_value, overall)
    mechanism = mechanism_assessment(events, overall, bootstrap)
    controls, control_bindings = historical_control_summary(Path(cli.result_root))

    artifacts = {
        "dataset_samples_train_valid.csv": samples,
        "dataset_split_summary.csv": split_summary,
        "dataset_video_summary.csv": videos,
        "modality_feature_sample_quality.csv": feature_samples,
        "modality_feature_summary.csv": feature_summary,
        "valid_prediction_events.csv": events,
        "overall_prediction_summary.csv": overall,
        "group_mechanism_summary.csv": groups,
        "modality_marginal_value.csv": modality_value,
        "video_bootstrap.csv": bootstrap,
        "opportunity_ranking.csv": opportunities,
        "source_metric_rows.csv": pd.DataFrame(source_metric_rows),
        "historical_control_valid_metrics.csv": controls,
    }
    for name, frame in artifacts.items():
        frame.to_csv(root / name, index=False)

    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": "experiment/cfcompat-distillation-evidence-v1",
        "formal_seeds": list(FORMAL_SEEDS),
        "splits_constructed": ["train", "valid"],
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "model_training_performed": False,
        "optimizer_constructed": False,
        "backward_called": False,
        "source_records": source_records,
        "historical_control_bindings": control_bindings,
        "artifacts": {},
    }
    for name in artifacts:
        path = root / name
        manifest["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    manifest_path = root / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": mechanism["verdict"],
        "dataset_shift": shift,
        "dataset_limitations": limitations,
        "mechanism_assessment": mechanism,
        "protocol": {
            "dataset": "mosi",
            "source_splits": "official_train_and_valid_only",
            "formal_seeds": list(FORMAL_SEEDS),
            "comparison": "validation_best_DLF_ModDrop_vs_validation_best_CFCompatKD",
            "valid_compatibility": "train_calibrated_counterfactual_shift_proxy_descriptive_only",
            "bootstrap_unit": "source_video_joint_across_seeds_and_modes",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "official_test_constructed": False,
            "official_test_authorized": False,
            "model_training_performed": False,
            "new_method_authorized": False,
        },
    }
    summary_path = root / "audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = root / "audit_report.md"
    report_path.write_text(
        render_report(summary, split_summary, overall, opportunities, controls),
        encoding="utf-8",
    )

    print("MOSI CFCompatKD dataset-mechanism audit complete")
    print("verdict:", mechanism["verdict"])
    print("mean two-seed Valid J gain: {:+.6f}".format(mechanism["mean_two_seed_J_gain"]))
    print(
        "video-bootstrap 95% CI: [{:+.6f}, {:+.6f}]".format(
            mechanism["video_bootstrap_J_gain"]["ci95_low"],
            mechanism["video_bootstrap_J_gain"]["ci95_high"],
        )
    )
    print("official Test was not constructed")
    print("report:", report_path)


if __name__ == "__main__":
    main()
