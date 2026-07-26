#!/usr/bin/env python3
"""Replay one frozen Expert checkpoint and extract its own static risk signals."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage23d_self_risk_common import (
    ACTIVE_HEADS,
    EXPERTS,
    HEADS,
    LEGAL_SUBMODES,
    MODE_MASKS,
    MODES,
    OUT,
    ROOT,
    ReadOnlyActivationCapture,
    active_head_features,
    atomic_json,
    cosine_and_distance,
    expert_directory,
    sha256_file,
    sha256_json,
    summary_features,
    utc_now,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.missing_utils import MissingModalityWrapper, mode_to_mask
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-fold", required=True, type=int, choices=(0, 1))
    parser.add_argument("--expert-id", required=True, choices=EXPERTS)
    parser.add_argument("--gpu-id", required=True, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--smoke-batches", default=0, type=int)
    return parser.parse_args()


def atomic_gzip_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed:
            frame.to_csv(compressed, index=False, float_format="%.10g")
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_args(cli):
    args = get_config_regression(
        "DLF", "mosei", str(ROOT / "config" / "config.json")
    )
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = False
    args.train_mode = "regression"
    args.batch_size = int(cli.batch_size)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def batch_flat(tensor, batch_size):
    value = tensor.detach().cpu()
    if value.size(0) == batch_size:
        return value.reshape(batch_size, -1).numpy()
    if value.ndim >= 2 and value.size(1) == batch_size:
        axes = [1, 0] + list(range(2, value.ndim))
        return value.permute(*axes).reshape(batch_size, -1).numpy()
    raise ValueError(f"Cannot identify batch dimension in {tuple(value.shape)}")


def per_sample_mse(left, right, batch_size):
    if (
        left.ndim == 3
        and right.ndim == 3
        and left.size(1) == batch_size
        and right.size(0) == batch_size
        and left.size(0) == right.size(2)
        and left.size(2) == right.size(1)
    ):
        left = left.permute(1, 2, 0)
    left_values = batch_flat(left, batch_size)
    right_values = batch_flat(right, batch_size)
    if left_values.shape != right_values.shape:
        raise ValueError(
            f"Residual tensors differ {left_values.shape} != {right_values.shape}"
        )
    return np.mean(np.square(left_values - right_values), axis=1)


def effective_lengths(text, audio, vision):
    text_cpu = text.detach().cpu()
    if text_cpu.ndim == 3 and text_cpu.size(1) >= 2:
        text_length = (text_cpu[:, 1, :] > 0).sum(dim=1).numpy()
    else:
        text_length = (
            torch.linalg.vector_norm(text_cpu.float(), dim=-1) > 0
        ).sum(dim=1).numpy()
    audio_length = (
        torch.linalg.vector_norm(audio.detach().cpu().float(), dim=-1) > 0
    ).sum(dim=1).numpy()
    vision_length = (
        torch.linalg.vector_norm(vision.detach().cpu().float(), dim=-1) > 0
    ).sum(dim=1).numpy()
    return text_length, audio_length, vision_length


def representation_features(mode, position, captured, output, batch_size):
    available = MODE_MASKS[mode]
    result = {}
    final_fused = batch_flat(captured["final_fused"], batch_size)[position]
    shared_fused = batch_flat(captured["shared_fused"], batch_size)[position]
    result.update(summary_features("final_fused", final_fused))
    result.update(summary_features("shared_fused", shared_fused))
    result.update(
        summary_features(
            "fusion_input",
            batch_flat(output["fusion_input"], batch_size)[position],
        )
    )
    lfa_names = ["lfa_l"]
    if available[1]:
        lfa_names.extend(["lfa_a", "lfa_cross_a", "lfa_ffn_a"])
    if available[2]:
        lfa_names.extend(["lfa_v", "lfa_cross_v", "lfa_ffn_v"])
    for name in lfa_names:
        result.update(
            summary_features(
                name, batch_flat(output[name], batch_size)[position]
            )
        )
    specific = {}
    aligned_shared = {}
    for name, flag in zip(("l", "a", "v"), available):
        if not flag:
            continue
        specific[name] = batch_flat(
            captured[f"specific_{name}"], batch_size
        )[position]
        aligned_shared[name] = batch_flat(
            output[f"c_{name}_sim"], batch_size
        )[position]
        result.update(summary_features(f"specific_{name}", specific[name]))
        result.update(
            summary_features(f"aligned_shared_{name}", aligned_shared[name])
        )
        result.update(
            cosine_and_distance(
                f"shared_specific_{name}",
                aligned_shared[name],
                specific[name],
            )
        )
    names = sorted(specific)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result.update(
                cosine_and_distance(
                    f"specific_pair_{left}_{right}",
                    specific[left],
                    specific[right],
                )
            )
    result["reconstruction_residual_mean"] = float(
        np.mean(
            [
                per_sample_mse(
                    output[f"recon_{name}"],
                    output[f"origin_{name}"],
                    batch_size,
                )[position]
                for name, flag in zip(("l", "a", "v"), available)
                if flag
            ]
        )
    )
    result["specific_reencode_residual_mean"] = float(
        np.mean(
            [
                per_sample_mse(
                    output[f"s_{name}"],
                    output[f"s_{name}_r"],
                    batch_size,
                )[position]
                for name, flag in zip(("l", "a", "v"), available)
                if flag
            ]
        )
    )
    return result


def raw_vector(mode, position, captured, output, batch_size):
    available = MODE_MASKS[mode]
    vectors = [
        batch_flat(captured["final_fused"], batch_size)[position],
        batch_flat(captured["shared_fused"], batch_size)[position],
        batch_flat(output["fusion_input"], batch_size)[position],
        batch_flat(output["lfa_l"], batch_size)[position],
    ]
    if available[1]:
        vectors.extend(
            [
                batch_flat(output[name], batch_size)[position]
                for name in ("lfa_a", "lfa_cross_a", "lfa_ffn_a")
            ]
        )
    if available[2]:
        vectors.extend(
            [
                batch_flat(output[name], batch_size)[position]
                for name in ("lfa_v", "lfa_cross_v", "lfa_ffn_v")
            ]
        )
    for name, flag in zip(("l", "a", "v"), available):
        if not flag:
            continue
        vectors.append(
            batch_flat(captured[f"specific_{name}"], batch_size)[position]
        )
        vectors.append(batch_flat(output[f"c_{name}_sim"], batch_size)[position])
    return np.concatenate(vectors).astype(np.float32, copy=False)


def submode_features(query_mode, position, values):
    legal = LEGAL_SUBMODES[query_mode]
    predictions = np.asarray(
        [values[mode]["heads"]["output_logit"][position] for mode in legal],
        dtype=np.float64,
    )
    query = predictions[0]
    sign_flip = np.sum(
        (np.sign(predictions[1:]) != np.sign(query)) & (predictions[1:] != 0)
    )
    availability = np.asarray([sum(MODE_MASKS[mode]) for mode in legal])
    if len(legal) > 1 and np.std(predictions) > 0:
        order_consistency = float(
            abs(np.corrcoef(availability, predictions)[0, 1])
        )
    else:
        order_consistency = 1.0
    head_variances = []
    for head in ACTIVE_HEADS[query_mode]:
        head_values = [
            values[mode]["heads"][head][position]
            for mode in legal
            if head in ACTIVE_HEADS[mode]
        ]
        if len(head_values) > 1:
            head_variances.append(np.var(head_values))
    return {
        "submode_count": len(legal),
        "submode_prediction_variance": float(np.var(predictions)),
        "submode_max_deviation": float(np.max(np.abs(predictions - query))),
        "submode_sign_flip_count": int(sign_flip),
        "submode_rank_order_consistency": order_consistency,
        "submode_hierarchical_head_variance": float(
            np.mean(head_variances) if head_variances else 0.0
        ),
    }


def main():
    cli = parse_args()
    protocol_path = OUT / "protocol" / "frozen_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol["internal_state_extraction_authorized"]:
        raise RuntimeError("Internal-state extraction is not authorized")
    if (
        protocol["expert_retraining_authorized"]
        or protocol["official_valid_authorized"]
        or protocol["test_authorized"]
        or protocol["stage23c_modification_authorized"]
    ):
        raise RuntimeError("Forbidden authorization is open")
    setup_seed(23800 + cli.checkpoint_fold)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args = build_args(cli)
    dataset = MMDataset(args, mode="train")
    args.seq_lens = dataset.get_seq_len()
    split = pd.read_csv(
        OUT / "protocol" / "sample_split_manifest.csv",
        dtype={"sample_id": str, "video_id": str},
    )
    holdout = split.loc[
        split["checkpoint_fold"].astype(int) == cli.checkpoint_fold
    ].copy()
    indices = holdout["train_index"].astype(int).tolist()
    role_by_id = holdout.set_index("sample_id")["self_risk_role"].to_dict()
    source_by_id = holdout.set_index("sample_id")["video_id"].to_dict()
    index_by_id = holdout.set_index("sample_id")["train_index"].astype(int).to_dict()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=cli.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cli.num_workers,
    )
    directory = expert_directory(cli.checkpoint_fold, cli.expert_id)
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA mismatch")
    model = MissingModalityWrapper(
        DLF(args), args.feature_dims[1], args.feature_dims[2]
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model.to(args.device).eval()
    expected = pd.read_csv(
        directory.parent.parent / "oof_predictions.csv"
    )
    expected = expected.loc[expected["expert_id"] == cli.expert_id].set_index(
        ["sample_id", "mode"]
    )["prediction"]
    rows = []
    raw = {
        mode: {"sample_id": [], "row_binding_sha256": [], "features": []}
        for mode in MODES
    }
    replay_differences = []
    prediction_before = {}
    prediction_after = {}
    with ReadOnlyActivationCapture(model) as capture, torch.no_grad():
        for batch_number, batch in enumerate(loader, 1):
            if cli.smoke_batches and batch_number > cli.smoke_batches:
                break
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            identifiers = [str(value) for value in batch["id"]]
            labels = batch["labels"]["M"].view(-1).numpy()
            text_length, audio_length, vision_length = effective_lengths(
                text, audio, vision
            )
            mode_values = {}
            for mode in MODES:
                capture.clear()
                mask = mode_to_mask(
                    mode, text.size(0), args.device, audio.dtype
                )
                output = model(text, audio, vision, mask)
                if set(capture.values) != set(capture.MODULES):
                    raise RuntimeError("Read-only hook did not capture all heads")
                heads = {
                    name: output[name].view(-1).detach().cpu().numpy()
                    for name in HEADS
                }
                mode_values[mode] = {
                    "output": output,
                    "captured": {
                        key: value.detach().clone()
                        for key, value in capture.values.items()
                    },
                    "heads": heads,
                }
                key = (identifiers[0], mode)
                prediction_before[key] = float(heads["output_logit"][0])
                prediction_after[key] = float(
                    output["output_logit"].view(-1)[0].detach().cpu()
                )
            batch_size = len(identifiers)
            for mode in MODES:
                values = mode_values[mode]
                output = values["output"]
                captured = values["captured"]
                for position, sample_id in enumerate(identifiers):
                    head_values = {
                        name: float(values["heads"][name][position])
                        for name in HEADS
                    }
                    old_prediction = float(expected.loc[(sample_id, mode)])
                    difference = abs(
                        head_values["output_logit"] - old_prediction
                    )
                    replay_differences.append(difference)
                    if difference > 1e-6:
                        raise RuntimeError(
                            f"Replay gate failed {sample_id}/{mode}: {difference}"
                        )
                    row_binding = sha256_json(
                        {
                            "sample_id": sample_id,
                            "mode": mode,
                            "expert_id": cli.expert_id,
                            "checkpoint_sha256": manifest["checkpoint_sha256"],
                        }
                    )
                    feature_row = {
                        "sample_id": sample_id,
                        "video_id": source_by_id[sample_id],
                        "train_index": index_by_id[sample_id],
                        "checkpoint_fold": cli.checkpoint_fold,
                        "expert_id": cli.expert_id,
                        "mode": mode,
                        "self_risk_role": role_by_id[sample_id],
                        "label": float(labels[position]),
                        "row_binding_sha256": row_binding,
                        "text_length": int(text_length[position]),
                        "audio_length": int(audio_length[position]),
                        "vision_length": int(vision_length[position]),
                        "mask_l": MODE_MASKS[mode][0],
                        "mask_a": MODE_MASKS[mode][1],
                        "mask_v": MODE_MASKS[mode][2],
                        **active_head_features(mode, head_values),
                        **representation_features(
                            mode, position, captured, output, batch_size
                        ),
                        **submode_features(mode, position, mode_values),
                    }
                    rows.append(feature_row)
                    raw[mode]["sample_id"].append(sample_id)
                    raw[mode]["row_binding_sha256"].append(row_binding)
                    raw[mode]["features"].append(
                        raw_vector(
                            mode, position, captured, output, batch_size
                        )
                    )
    if prediction_before != prediction_after:
        raise RuntimeError("Read-only hook changed final predictions")
    frame = pd.DataFrame(rows)
    expected_samples = (
        min(len(indices), cli.smoke_batches * cli.batch_size)
        if cli.smoke_batches
        else len(indices)
    )
    if (
        len(frame) != expected_samples * len(MODES)
        or frame.duplicated(["sample_id", "mode"]).any()
    ):
        raise RuntimeError("Static feature ledger binding/completeness failure")
    group = "smoke" if cli.smoke_batches else "features"
    output_dir = (
        OUT
        / group
        / f"checkpoint_fold{cli.checkpoint_fold}"
        / cli.expert_id
    )
    feature_artifacts = {}
    for mode in MODES:
        local = frame.loc[frame["mode"] == mode].dropna(axis=1, how="all")
        if local.isna().any().any():
            raise RuntimeError(f"Missing value in {mode} feature ledger")
        feature_local = local.drop(columns=["label"])
        numeric = feature_local.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise RuntimeError(f"NaN/Inf in {mode} feature ledger")
        feature_path = output_dir / f"static_internal_features_{mode}.csv.gz"
        atomic_gzip_csv(feature_local, feature_path)
        development_label_path = (
            output_dir / f"development_risk_labels_{mode}.csv.gz"
        )
        atomic_gzip_csv(
            local.loc[
                local["self_risk_role"] != "outer",
                [
                    "sample_id",
                    "video_id",
                    "mode",
                    "self_risk_role",
                    "label",
                    "row_binding_sha256",
                ],
            ],
            development_label_path,
        )
        sealed_outer_label_path = (
            output_dir / "sealed_outer_labels" / f"outer_labels_{mode}.csv.gz"
        )
        atomic_gzip_csv(
            local.loc[
                local["self_risk_role"] == "outer",
                [
                    "sample_id",
                    "video_id",
                    "mode",
                    "label",
                    "row_binding_sha256",
                ],
            ],
            sealed_outer_label_path,
        )
        feature_artifacts[mode] = {
            "path": str(feature_path.resolve()),
            "sha256": sha256_file(feature_path),
            "rows": len(local),
            "columns": len(feature_local.columns),
            "contains_label": False,
            "development_label_path": str(development_label_path.resolve()),
            "development_label_sha256": sha256_file(development_label_path),
            "sealed_outer_label_path": str(sealed_outer_label_path.resolve()),
            "sealed_outer_label_sha256": sha256_file(sealed_outer_label_path),
        }
    binding_path = output_dir / "feature_binding_ledger.csv.gz"
    atomic_gzip_csv(
        frame[
            [
                "sample_id",
                "video_id",
                "train_index",
                "checkpoint_fold",
                "expert_id",
                "mode",
                "self_risk_role",
                "prediction",
                "row_binding_sha256",
            ]
        ],
        binding_path,
    )
    raw_paths = {}
    for mode in MODES:
        vectors = np.stack(raw[mode]["features"])
        path = output_dir / f"raw_internal_vectors_{mode}.npz"
        atomic_npz(
            path,
            sample_id=np.asarray(raw[mode]["sample_id"]),
            row_binding_sha256=np.asarray(raw[mode]["row_binding_sha256"]),
            features=vectors,
        )
        raw_paths[mode] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": vectors.shape[0],
            "dimensions": vectors.shape[1],
        }
    extraction_manifest = {
        "stage": "Stage23D-A frozen Expert static internal-state extraction",
        "status": "COMPLETED",
        "feature_schema_version": 2,
        "checkpoint_fold": cli.checkpoint_fold,
        "expert_id": cli.expert_id,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "run_manifest_sha256": sha256_file(manifest_path),
        "feature_artifacts": feature_artifacts,
        "feature_binding_path": str(binding_path.resolve()),
        "feature_binding_sha256": sha256_file(binding_path),
        "rows": len(frame),
        "samples": frame["sample_id"].nunique(),
        "modes": list(MODES),
        "raw_vector_artifacts": raw_paths,
        "max_abs_final_replay_diff": float(max(replay_differences)),
        "hook_prediction_max_abs_diff": float(
            max(
                abs(prediction_before[key] - prediction_after[key])
                for key in prediction_before
            )
        ),
        "duplicates": int(frame.duplicated(["sample_id", "mode"]).sum()),
        "missing": 0,
        "nan_inf_count": 0,
        "inactive_head_leakage": 0,
        "source_split_sha256": protocol["source_split_sha256"],
        "expert_retrained": False,
        "student_trained": False,
        "arbiter_trained": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "gpu_id": cli.gpu_id,
        "completed_at": utc_now(),
    }
    atomic_json(output_dir / "extraction_manifest.json", extraction_manifest)
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(extraction_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
