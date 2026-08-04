"""Run the analysis-only CFCompatKD seed-mechanism audit.

The audit consumes the already-frozen Adam and selected-SAM Valid artifacts,
uses a deterministic train-only gradient probe, and never constructs Test.
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
from torch.utils.data import DataLoader, Subset

from data_loader import MMDataset
from train_cf_compat_kd import batch_to_device, build_config
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    build_frozen_evaluator,
    compatibility_for_modes,
    evaluator_prediction,
    gate_weights,
    gated_kd_loss,
)
from trains.singleTask.cfcompat_mechanism_utils import (
    FORMAL_SEEDS,
    MAX_SAMPLES_PER_VIDEO,
    METHOD,
    MISSING_MODES,
    MODES,
    OBJECTIVE_PAIRS,
    OUTPUT_TAG,
    PARAMETER_GROUPS,
    PROBE_SAMPLE_COUNT,
    VERSION,
    calibration_stats,
    gradient_pair_stats,
    infer_mechanism_flags,
    intensity_bin,
    label_bin,
    parameter_groups,
    parse_video_id,
    pearson,
    rank_compatibility_proxy,
    select_probe_positions,
    spearman,
    summarize_mode_deltas,
    tensor_state_sha256,
)
from trains.singleTask.fixed_kd_utils import (
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analysis-only CFCompatKD seed mechanism audit."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FORMAL_SEEDS))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--probe-samples", type=int, default=PROBE_SAMPLE_COUNT)
    args = parser.parse_args()
    if tuple(args.seeds) != FORMAL_SEEDS:
        parser.error("Mechanism audit fixes seeds to 1111 1114.")
    if args.num_workers != 1:
        parser.error("Mechanism audit fixes num_workers=1.")
    if args.probe_samples != PROBE_SAMPLE_COUNT:
        parser.error("Mechanism audit fixes probe samples to 64.")
    return args


def output_paths(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "valid_train_audit"
    )
    root.mkdir(parents=True, exist_ok=True)
    sam_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_sam_v1"
        / cli.dataset
        / "valid_screen"
    )
    return root, sam_root


def resolve_path(value, result_root):
    path = Path(str(value))
    if path.is_file():
        return path
    if not path.is_absolute():
        candidates = [
            Path.cwd() / path,
            Path(result_root).resolve().parent / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Required artifact is absent: {}".format(value))


def load_sam_artifacts(cli, sam_root):
    paths = {
        "summary": sam_root / "sam_valid_screen_summary.json",
        "grid": sam_root / "sam_valid_grid_summary.csv",
        "predictions": sam_root / "sam_all_valid_predictions.csv",
        "audit": sam_root / "sam_valid_screen_audit_check.json",
        "source": sam_root / "sam_source_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing SAM source artifacts:\n" + "\n".join(missing))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    grid = pd.read_csv(paths["grid"])
    predictions = pd.read_csv(paths["predictions"])
    if not audit.get("passed", False):
        raise RuntimeError("SAM source audit did not pass.")
    if tuple(summary["protocol"]["formal_seeds"]) != FORMAL_SEEDS:
        raise RuntimeError("SAM source seeds differ from the mechanism protocol.")
    if summary["protocol"]["official_test_constructed"]:
        raise RuntimeError("Mechanism audit refuses a source that constructed Test.")
    if predictions.sample_index.nunique() != 229:
        raise RuntimeError("SAM Valid predictions must contain 229 unique samples.")
    if grid.TestConstructed.map(
        lambda value: str(value).strip().lower() == "true"
    ).any():
        raise RuntimeError("SAM grid unexpectedly contains Test results.")
    selected_rho = float(summary["selected_rho"])
    selected_run = "sam_rho_{}".format(str(selected_rho).replace(".", "p"))
    return {
        "paths": paths,
        "summary": summary,
        "audit": audit,
        "source": source,
        "grid": grid,
        "predictions": predictions,
        "selected_rho": selected_rho,
        "selected_run": selected_run,
    }


def source_row(source, seed, run):
    local = source["grid"].loc[
        source["grid"].Seed.astype(int).eq(int(seed))
        & source["grid"].Run.astype(str).eq(str(run))
    ]
    if len(local) != 1:
        raise RuntimeError(
            "Source grid row is not unique for seed={} run={}".format(seed, run)
        )
    return local.iloc[0].to_dict()


def load_student(args, checkpoint):
    backbone = DLF(args).to(args.device)
    student = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    return student


def train_metadata(args):
    dataset = MMDataset(args, mode="train")
    rows = []
    for position in range(len(dataset)):
        sample = dataset[position]
        sample_id = str(sample["id"])
        rows.append({
            "dataset_position": int(position),
            "sample_index": int(sample["index"]),
            "sample_id": sample_id,
            "video_id": parse_video_id(sample_id),
            "label": float(sample["labels"]["M"].view(-1)[0].item()),
        })
    return dataset, pd.DataFrame(rows)


def build_probe(cli, args, output_root):
    dataset, metadata = train_metadata(args)
    selected = select_probe_positions(
        metadata,
        target_count=cli.probe_samples,
        max_per_video=MAX_SAMPLES_PER_VIDEO,
    )
    selected.to_csv(output_root / "gradient_probe_manifest.csv", index=False)
    positions = selected.sort_values(
        "selection_rank", kind="mergesort"
    ).dataset_position.astype(int).tolist()
    loader = DataLoader(
        Subset(dataset, positions),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cli.num_workers,
    )
    return loader, selected


def valid_context(cli, source, seed):
    setup_seed(seed)
    args = build_config(cli, seed)
    valid_loader = build_single_split_loader(args, "valid", cli.num_workers)
    baseline = source_row(source, seed, "adam_replay")
    teacher_checkpoint = resolve_path(
        baseline["TeacherCheckpoint"], cli.result_root
    )
    evaluator_checkpoint = resolve_path(
        baseline["EvaluatorCheckpoint"], cli.result_root
    )
    teacher = build_frozen_teacher(DLF, args, teacher_checkpoint)
    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    rows = []
    teacher.eval()
    evaluator.eval()
    for batch in valid_loader:
        text, audio, vision, labels = batch_to_device(batch, args.device)
        teacher_pred = teacher_lav_prediction(
            teacher, text, audio, vision
        ).view(-1).cpu().numpy()
        evaluator_preds = {
            mode: evaluator_prediction(
                evaluator, text, audio, vision, mode
            ).view(-1).cpu().numpy()
            for mode in MODES
        }
        indices = batch["index"].view(-1).cpu().numpy().astype(int)
        ids = list(batch["id"])
        label_values = labels.view(-1).cpu().numpy()
        for offset, index in enumerate(indices):
            row = {
                "Seed": int(seed),
                "sample_index": int(index),
                "sample_id": str(ids[offset]),
                "teacher_LAV_pred": float(teacher_pred[offset]),
                "teacher_abs_error": float(
                    abs(teacher_pred[offset] - label_values[offset])
                ),
            }
            for mode in MODES:
                row["evaluator_{}_pred".format(mode)] = float(
                    evaluator_preds[mode][offset]
                )
            for mode in MISSING_MODES:
                row["evaluator_delta_{}".format(mode)] = float(
                    abs(
                        evaluator_preds[mode][offset]
                        - evaluator_preds["LAV"][offset]
                    )
                )
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        "sample_index", kind="mergesort"
    )
    if len(frame) != 229 or frame.sample_index.nunique() != 229:
        raise RuntimeError("Valid context did not contain 229 unique samples.")
    for mode in MISSING_MODES:
        frame["valid_compat_proxy_{}".format(mode)] = (
            rank_compatibility_proxy(
                frame["evaluator_delta_{}".format(mode)].to_numpy(dtype=float)
            )
        )
    del teacher, evaluator, valid_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return frame


def prediction_audit(cli, source, output_root):
    predictions = source["predictions"].copy()
    all_sample_rows = []
    all_mode_rows = []
    context_rows = []
    for seed in FORMAL_SEEDS:
        context = valid_context(cli, source, seed)
        context_rows.append(context)
        base = predictions.loc[
            predictions.Seed.astype(int).eq(seed)
            & predictions.Run.astype(str).eq("adam_replay")
        ].copy()
        sam = predictions.loc[
            predictions.Seed.astype(int).eq(seed)
            & predictions.Run.astype(str).eq(source["selected_run"])
        ].copy()
        if len(base) != 229 or len(sam) != 229:
            raise RuntimeError(
                "Seed {} source runs do not each contain 229 rows.".format(seed)
            )
        keys = ["sample_index", "sample_id", "label"]
        merged = base[keys + [
            "{}_pred".format(mode) for mode in MODES
        ]].merge(
            sam[keys + ["{}_pred".format(mode) for mode in MODES]],
            on=keys,
            how="inner",
            suffixes=("_baseline", "_sam"),
            validate="one_to_one",
        )
        merged = merged.merge(
            context,
            on=["sample_index", "sample_id"],
            how="left",
            validate="one_to_one",
        )
        if len(merged) != 229 or merged.isna().any().any():
            raise RuntimeError("Prediction/context merge is incomplete.")
        merged["Seed"] = int(seed)
        merged["video_id"] = merged.sample_id.map(parse_video_id)
        merged["label_bin"] = merged.label.map(label_bin)
        merged["intensity_bin"] = merged.label.map(intensity_bin)

        for mode in MODES:
            base_pred = merged["{}_pred_baseline".format(mode)].to_numpy(dtype=float)
            sam_pred = merged["{}_pred_sam".format(mode)].to_numpy(dtype=float)
            labels = merged.label.to_numpy(dtype=float)
            merged["baseline_{}_abs_error".format(mode)] = np.abs(
                base_pred - labels
            )
            merged["sam_{}_abs_error".format(mode)] = np.abs(sam_pred - labels)
            merged["delta_{}_abs_error".format(mode)] = (
                merged["sam_{}_abs_error".format(mode)]
                - merged["baseline_{}_abs_error".format(mode)]
            )
            merged["prediction_change_{}".format(mode)] = sam_pred - base_pred
            for row in merged.itertuples(index=False):
                data = row._asdict()
                all_mode_rows.append({
                    "Seed": int(seed),
                    "sample_index": int(data["sample_index"]),
                    "sample_id": str(data["sample_id"]),
                    "video_id": str(data["video_id"]),
                    "label": float(data["label"]),
                    "label_bin": str(data["label_bin"]),
                    "intensity_bin": str(data["intensity_bin"]),
                    "Mode": mode,
                    "baseline_prediction": float(
                        data["{}_pred_baseline".format(mode)]
                    ),
                    "sam_prediction": float(
                        data["{}_pred_sam".format(mode)]
                    ),
                    "baseline_abs_error": float(
                        data["baseline_{}_abs_error".format(mode)]
                    ),
                    "sam_abs_error": float(
                        data["sam_{}_abs_error".format(mode)]
                    ),
                    "delta_abs_error": float(
                        data["delta_{}_abs_error".format(mode)]
                    ),
                    "prediction_change": float(
                        data["prediction_change_{}".format(mode)]
                    ),
                    "teacher_abs_error": float(data["teacher_abs_error"]),
                    "teacher_LAV_pred": float(data["teacher_LAV_pred"]),
                    "evaluator_delta": (
                        float(data["evaluator_delta_{}".format(mode)])
                        if mode in MISSING_MODES else 0.0
                    ),
                    "valid_compat_proxy": (
                        float(data["valid_compat_proxy_{}".format(mode)])
                        if mode in MISSING_MODES else 1.0
                    ),
                })

        merged["baseline_J_proxy"] = (
            0.5 * merged["baseline_LAV_abs_error"]
            + (1.0 / 6.0)
            * sum(
                merged["baseline_{}_abs_error".format(mode)]
                for mode in MISSING_MODES
            )
        )
        merged["sam_J_proxy"] = (
            0.5 * merged["sam_LAV_abs_error"]
            + (1.0 / 6.0)
            * sum(
                merged["sam_{}_abs_error".format(mode)]
                for mode in MISSING_MODES
            )
        )
        merged["delta_J_proxy"] = (
            merged.sam_J_proxy - merged.baseline_J_proxy
        )
        merged["baseline_teacher_gap_LAV"] = np.abs(
            merged.LAV_pred_baseline - merged.teacher_LAV_pred
        )
        merged["sam_teacher_gap_LAV"] = np.abs(
            merged.LAV_pred_sam - merged.teacher_LAV_pred
        )
        all_sample_rows.append(merged)

    sample_rows = pd.concat(all_sample_rows, ignore_index=True)
    mode_rows = pd.DataFrame(all_mode_rows)
    context_frame = pd.concat(context_rows, ignore_index=True)
    sample_rows.to_csv(output_root / "prediction_sample_deltas.csv", index=False)
    mode_rows.to_csv(output_root / "prediction_mode_deltas.csv", index=False)
    context_frame.to_csv(output_root / "valid_teacher_evaluator_context.csv", index=False)

    mode_summary = summarize_mode_deltas(mode_rows)
    mode_summary.to_csv(output_root / "prediction_mode_summary.csv", index=False)

    group_rows = []
    for group_type, column in (
        ("label_bin", "label_bin"),
        ("intensity_bin", "intensity_bin"),
    ):
        for keys, local in mode_rows.groupby(
            ["Seed", "Mode", column], sort=True
        ):
            group_rows.append({
                "Seed": int(keys[0]),
                "Mode": str(keys[1]),
                "group_type": group_type,
                "group": str(keys[2]),
                "count": int(len(local)),
                "mean_delta_abs_error": float(local.delta_abs_error.mean()),
                "improved_fraction": float((local.delta_abs_error < 0).mean()),
            })
    pd.DataFrame(group_rows).to_csv(
        output_root / "prediction_group_summary.csv", index=False
    )

    video_rows = []
    for keys, local in sample_rows.groupby(["Seed", "video_id"], sort=True):
        video_rows.append({
            "Seed": int(keys[0]),
            "video_id": str(keys[1]),
            "count": int(len(local)),
            "mean_delta_J_proxy": float(local.delta_J_proxy.mean()),
            "improved_fraction": float((local.delta_J_proxy < 0).mean()),
        })
    pd.DataFrame(video_rows).to_csv(
        output_root / "prediction_video_summary.csv", index=False
    )

    seed_frames = {
        seed: sample_rows.loc[
            sample_rows.Seed.astype(int).eq(seed),
            ["sample_id", "delta_J_proxy"],
        ].rename(columns={"delta_J_proxy": "delta_J_seed{}".format(seed)})
        for seed in FORMAL_SEEDS
    }
    cross = seed_frames[1111].merge(
        seed_frames[1114], on="sample_id", validate="one_to_one"
    )
    left = cross.delta_J_seed1111.to_numpy(dtype=float)
    right = cross.delta_J_seed1114.to_numpy(dtype=float)
    cross["same_direction"] = np.sign(left) == np.sign(right)
    cross["both_improve"] = (left < 0) & (right < 0)
    cross["opposite_direction"] = np.sign(left) != np.sign(right)
    cross.to_csv(
        output_root / "cross_seed_sample_comparison.csv", index=False
    )
    cross_summary = {
        "count": int(len(cross)),
        "pearson_delta_J_proxy": pearson(left, right),
        "spearman_delta_J_proxy": spearman(left, right),
        "same_direction_fraction": float(cross.same_direction.mean()),
        "both_improve_fraction": float(cross.both_improve.mean()),
        "opposite_direction_fraction": float(cross.opposite_direction.mean()),
    }

    calibration_rows = []
    shrinkage_rows = []
    for seed in FORMAL_SEEDS:
        local = sample_rows.loc[sample_rows.Seed.astype(int).eq(seed)]
        for mode in MODES:
            baseline_stats = calibration_stats(
                local.label, local["{}_pred_baseline".format(mode)]
            )
            sam_stats = calibration_stats(
                local.label, local["{}_pred_sam".format(mode)]
            )
            for variant, stats in (
                ("baseline", baseline_stats),
                ("sam", sam_stats),
            ):
                calibration_rows.append({
                    "Seed": int(seed),
                    "Mode": mode,
                    "Variant": variant,
                    **stats,
                })
            shrinkage_rows.append({
                "Seed": int(seed),
                "Mode": mode,
                "prediction_std_ratio_sam_to_baseline": (
                    sam_stats["prediction_std"]
                    / max(baseline_stats["prediction_std"], 1e-12)
                ),
                "prediction_abs_mean_ratio_sam_to_baseline": (
                    sam_stats["prediction_abs_mean"]
                    / max(baseline_stats["prediction_abs_mean"], 1e-12)
                ),
                "slope_change_sam_minus_baseline": (
                    sam_stats["slope_pred_on_label"]
                    - baseline_stats["slope_pred_on_label"]
                ),
                "mae_change_sam_minus_baseline": (
                    sam_stats["mae"] - baseline_stats["mae"]
                ),
            })
    calibration = pd.DataFrame(calibration_rows)
    shrinkage = pd.DataFrame(shrinkage_rows)
    calibration.to_csv(
        output_root / "prediction_calibration.csv", index=False
    )
    shrinkage.to_csv(
        output_root / "prediction_shrinkage.csv", index=False
    )

    relation_rows = []
    for seed in FORMAL_SEEDS:
        local_sample = sample_rows.loc[
            sample_rows.Seed.astype(int).eq(seed)
        ]
        relation_rows.extend([
            {
                "Seed": int(seed),
                "Mode": "J_proxy",
                "Relation": "delta_vs_teacher_abs_error",
                "Pearson": pearson(
                    local_sample.delta_J_proxy,
                    local_sample.teacher_abs_error,
                ),
                "Spearman": spearman(
                    local_sample.delta_J_proxy,
                    local_sample.teacher_abs_error,
                ),
            },
            {
                "Seed": int(seed),
                "Mode": "LAV",
                "Relation": "delta_vs_baseline_teacher_gap",
                "Pearson": pearson(
                    local_sample.delta_LAV_abs_error,
                    local_sample.baseline_teacher_gap_LAV,
                ),
                "Spearman": spearman(
                    local_sample.delta_LAV_abs_error,
                    local_sample.baseline_teacher_gap_LAV,
                ),
            },
        ])
        local_modes = mode_rows.loc[
            mode_rows.Seed.astype(int).eq(seed)
            & mode_rows.Mode.isin(MISSING_MODES)
        ]
        for mode in MISSING_MODES:
            local = local_modes.loc[local_modes.Mode.eq(mode)]
            relation_rows.append({
                "Seed": int(seed),
                "Mode": mode,
                "Relation": "delta_vs_evaluator_delta",
                "Pearson": pearson(local.delta_abs_error, local.evaluator_delta),
                "Spearman": spearman(local.delta_abs_error, local.evaluator_delta),
            })
            relation_rows.append({
                "Seed": int(seed),
                "Mode": mode,
                "Relation": "delta_vs_valid_compat_proxy",
                "Pearson": pearson(
                    local.delta_abs_error, local.valid_compat_proxy
                ),
                "Spearman": spearman(
                    local.delta_abs_error, local.valid_compat_proxy
                ),
            })
    relations = pd.DataFrame(relation_rows)
    relations.to_csv(
        output_root / "prediction_context_relations.csv", index=False
    )
    return {
        "sample_rows": sample_rows,
        "mode_rows": mode_rows,
        "mode_summary": mode_summary,
        "cross_summary": cross_summary,
        "calibration": calibration,
        "shrinkage": shrinkage,
        "relations": relations,
    }


def load_train_assets(cli, args, row):
    teacher_checkpoint = resolve_path(row["TeacherCheckpoint"], cli.result_root)
    cache_path = resolve_path(row["CompatibilityCache"], cli.result_root)
    cache_frame = pd.read_csv(cache_path)
    cache_by_index = {
        int(record["sample_index"]): record
        for record in cache_frame.to_dict("records")
    }
    if len(cache_by_index) != 1284:
        raise RuntimeError("Train compatibility cache is not 1284 unique samples.")
    teacher = build_frozen_teacher(DLF, args, teacher_checkpoint)
    return teacher, cache_by_index


def objective_losses(
    model,
    teacher,
    cache_by_index,
    batch,
    mode,
    args,
    rng_seed,
    criterion,
    cosine,
    hinge,
):
    setup_seed(int(rng_seed))
    text, audio, vision, labels = batch_to_device(batch, args.device)
    initial_rng = torch.get_rng_state().clone()
    cuda_rng = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else []
    )
    full_mask = mode_to_mask(
        "LAV", labels.size(0), args.device, audio.dtype
    )
    full_output = model(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(
        full_output, labels, criterion, cosine, hinge
    )
    torch.set_rng_state(initial_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    missing_mask = mode_to_mask(
        mode, labels.size(0), args.device, audio.dtype
    )
    missing_output = model(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(
        missing_output, labels, criterion
    )
    teacher_prediction = teacher_lav_prediction(
        teacher, text, audio, vision
    )
    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    compatibility = compatibility_for_modes(
        cache_by_index,
        indices,
        [mode] * len(indices),
        args.device,
        labels.dtype,
    )
    gate, _ = gate_weights(
        compatibility, teacher_prediction, labels, "compat"
    )
    kd_loss, _ = gated_kd_loss(
        missing_output["output_logit"], teacher_prediction, gate
    )
    return {
        "full": full_loss,
        "missing": missing_loss,
        "kd": kd_loss,
    }


def gradient_audit(cli, source, output_root):
    gradient_rows = []
    state_rows = []
    group_manifest_rows = []
    probe_reference = None
    for seed in FORMAL_SEEDS:
        setup_seed(seed)
        args = build_config(cli, seed)
        probe_loader, probe_manifest = build_probe(
            cli, args, output_root
        )
        if probe_reference is None:
            probe_reference = probe_manifest[
                ["sample_index", "sample_id", "label", "video_id"]
            ].copy()
        else:
            current = probe_manifest[
                ["sample_index", "sample_id", "label", "video_id"]
            ].copy()
            if not current.equals(probe_reference):
                raise RuntimeError("Probe changed across formal seeds.")
        variants = (
            ("baseline", "adam_replay"),
            ("sam", source["selected_run"]),
        )
        for variant, run in variants:
            row = source_row(source, seed, run)
            checkpoint = resolve_path(row["MainCheckpoint"], cli.result_root)
            model = load_student(args, checkpoint)
            teacher, cache_by_index = load_train_assets(
                cli, args, row
            )
            model.train()
            groups = parameter_groups(model)
            for group_name, values in groups.items():
                group_manifest_rows.append({
                    "Seed": int(seed),
                    "Variant": variant,
                    "ParameterGroup": group_name,
                    "parameter_count": int(len(values)),
                    "parameter_numel": int(
                        sum(parameter.numel() for _, parameter in values)
                    ),
                })
            before_sha = tensor_state_sha256(model)
            criterion = nn.L1Loss()
            cosine = nn.CosineEmbeddingLoss()
            hinge = HingeLoss()
            for batch_index, batch in enumerate(probe_loader):
                for mode in MISSING_MODES:
                    losses = objective_losses(
                        model,
                        teacher,
                        cache_by_index,
                        batch,
                        mode,
                        args,
                        rng_seed=(
                            91000000
                            + int(seed) * 100
                            + int(batch_index)
                        ),
                        criterion=criterion,
                        cosine=cosine,
                        hinge=hinge,
                    )
                    group_items = list(groups.items())
                    for group_index, (group_name, named_params) in enumerate(
                        group_items
                    ):
                        params = [parameter for _, parameter in named_params]
                        gradients = {}
                        objective_names = ("full", "missing", "kd")
                        for objective_index, objective in enumerate(
                            objective_names
                        ):
                            is_last = (
                                group_index == len(group_items) - 1
                                and objective_index == len(objective_names) - 1
                            )
                            gradients[objective] = torch.autograd.grad(
                                losses[objective],
                                params,
                                retain_graph=not is_last,
                                create_graph=False,
                                allow_unused=True,
                            )
                        for left, right in OBJECTIVE_PAIRS:
                            stats = gradient_pair_stats(
                                gradients[left], gradients[right]
                            )
                            gradient_rows.append({
                                "Seed": int(seed),
                                "Variant": variant,
                                "Run": run,
                                "Mode": mode,
                                "ProbeBatch": int(batch_index),
                                "ParameterGroup": group_name,
                                "Pair": "{}_vs_{}".format(left, right),
                                "full_loss": float(
                                    losses["full"].detach().cpu()
                                ),
                                "missing_loss": float(
                                    losses["missing"].detach().cpu()
                                ),
                                "kd_loss": float(
                                    losses["kd"].detach().cpu()
                                ),
                                **stats,
                            })
                        del gradients
                    del losses
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            after_sha = tensor_state_sha256(model)
            state_rows.append({
                "Seed": int(seed),
                "Variant": variant,
                "Run": run,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256(checkpoint),
                "state_sha_before": before_sha,
                "state_sha_after": after_sha,
                "parameters_unchanged": bool(before_sha == after_sha),
            })
            del model, teacher
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del probe_loader

    raw = pd.DataFrame(gradient_rows)
    raw.to_csv(output_root / "gradient_pairwise_raw.csv", index=False)
    pd.DataFrame(state_rows).to_csv(
        output_root / "gradient_model_state_audit.csv", index=False
    )
    pd.DataFrame(group_manifest_rows).drop_duplicates().to_csv(
        output_root / "gradient_parameter_groups.csv", index=False
    )

    summary_rows = []
    for keys, local in raw.groupby(
        ["Seed", "Variant", "Mode", "ParameterGroup", "Pair"],
        sort=True,
    ):
        finite_cos = local.cosine[np.isfinite(local.cosine.astype(float))]
        summary_rows.append({
            "Seed": int(keys[0]),
            "Variant": str(keys[1]),
            "Mode": str(keys[2]),
            "ParameterGroup": str(keys[3]),
            "Pair": str(keys[4]),
            "batch_count": int(len(local)),
            "finite_cosine_count": int(len(finite_cos)),
            "mean_cosine": (
                float(finite_cos.astype(float).mean())
                if len(finite_cos) else float("nan")
            ),
            "median_cosine": (
                float(finite_cos.astype(float).median())
                if len(finite_cos) else float("nan")
            ),
            "conflict_fraction": (
                float((finite_cos.astype(float) < 0).mean())
                if len(finite_cos) else float("nan")
            ),
            "mean_left_norm": float(local.left_norm.mean()),
            "mean_right_norm": float(local.right_norm.mean()),
            "all_finite": bool(local.finite.all()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_root / "gradient_pair_summary.csv", index=False
    )

    baseline = summary.loc[
        summary.Variant.eq("baseline")
    ].drop(columns=["Variant"])
    sam = summary.loc[
        summary.Variant.eq("sam")
    ].drop(columns=["Variant"])
    keys = ["Seed", "Mode", "ParameterGroup", "Pair"]
    delta = baseline.merge(
        sam, on=keys, suffixes=("_baseline", "_sam"), validate="one_to_one"
    )
    delta["mean_cosine_change_sam_minus_baseline"] = (
        delta.mean_cosine_sam - delta.mean_cosine_baseline
    )
    delta["conflict_fraction_change_sam_minus_baseline"] = (
        delta.conflict_fraction_sam - delta.conflict_fraction_baseline
    )
    delta.to_csv(
        output_root / "gradient_sam_minus_baseline.csv", index=False
    )
    return {
        "raw": raw,
        "summary": summary,
        "delta": delta,
        "state": pd.DataFrame(state_rows),
    }


def render_report(summary):
    flags = summary["mechanism_flags"]
    lines = [
        "# CFCompatKD seed mechanism audit v1",
        "",
        "## Scope",
        "",
        "- Official Test constructed: `False`",
        "- New model training: `False`",
        "- Optimizer steps: `0`",
        "- Seeds: `1111, 1114`",
        "- Compared checkpoints: original Adam replay and selected SAM checkpoint",
        "",
        "## Evidence",
        "",
        "- Cross-seed sample delta Spearman: `{:.6f}`".format(
            summary["cross_seed_sample_effect"]["spearman_delta_J_proxy"]
        ),
        "- Same-direction sample fraction: `{:.6f}`".format(
            summary["cross_seed_sample_effect"]["same_direction_fraction"]
        ),
        "- Prediction shrinkage supported: `{}`".format(
            flags["prediction_shrinkage_supported"]
        ),
        "- Sample effect consistent across seeds: `{}`".format(
            flags["sample_effect_consistent_across_seeds"]
        ),
        "- SAM gradient-conflict change consistent: `{}`".format(
            flags["sam_gradient_conflict_change_consistent"]
        ),
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- Training authorization: `none`",
        "- Next action: inspect this audit before defining any new method.",
        "",
    ]
    return "\n".join(lines)


def main():
    cli = parse_args()
    output_root, sam_root = output_paths(cli)
    source = load_sam_artifacts(cli, sam_root)
    prediction = prediction_audit(cli, source, output_root)
    gradient = gradient_audit(cli, source, output_root)
    flags = infer_mechanism_flags(
        prediction["cross_summary"],
        prediction["shrinkage"],
        gradient["delta"],
    )
    if flags["sample_effect_consistent_across_seeds"]:
        verdict = "SHARED_SAMPLE_EFFECT_FOUND_REVIEW_MECHANISM"
    elif flags["prediction_shrinkage_supported"]:
        verdict = "PREDICTION_SHRINKAGE_FOUND_SEED_EFFECT_STILL_INCONSISTENT"
    elif flags["sam_gradient_conflict_change_consistent"]:
        verdict = "CONSISTENT_GRADIENT_CHANGE_FOUND_REVIEW_MECHANISM"
    else:
        verdict = "SEED_SPECIFIC_EFFECT_NO_SHARED_MECHANISM"

    source_manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_branch": "feature/cfcompat-sam-valid-screen-v1",
        "formal_seeds": list(FORMAL_SEEDS),
        "selected_rho": float(source["selected_rho"]),
        "sam_source_files": {
            name: {
                "path": str(path),
                "sha256": checkpoint_sha256(path),
            }
            for name, path in source["paths"].items()
        },
        "probe_sample_count": int(cli.probe_samples),
        "probe_max_samples_per_video": MAX_SAMPLES_PER_VIDEO,
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "new_model_training": False,
        "optimizer_steps": 0,
        "additional_inference_parameters": 0,
    }
    (
        output_root / "mechanism_source_manifest.json"
    ).write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "selected_rho": float(source["selected_rho"]),
        "cross_seed_sample_effect": prediction["cross_summary"],
        "mechanism_flags": flags,
        "gradient_models_unchanged": bool(
            gradient["state"].parameters_unchanged.all()
        ),
        "protocol": {
            "official_valid_predictions_only": True,
            "gradient_probe_split": "train",
            "official_test_constructed": False,
            "test_loader_construction_count": 0,
            "test_loader_traversal_count": 0,
            "new_model_training": False,
            "optimizer_steps": 0,
            "parameter_groups": list(PARAMETER_GROUPS),
            "probe_sample_count": int(cli.probe_samples),
            "max_samples_per_video": MAX_SAMPLES_PER_VIDEO,
            "valid_compatibility_is_descriptive_proxy_not_training_gate": True,
            "training_authorized": False,
        },
    }
    (
        output_root / "mechanism_summary.json"
    ).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (
        output_root / "mechanism_report.md"
    ).write_text(render_report(summary), encoding="utf-8")

    print("CFCompatKD seed mechanism audit complete")
    print("verdict:", verdict)
    print("official Test constructed: False")
    print("new model training: False")
    print("output:", output_root)


if __name__ == "__main__":
    main()
