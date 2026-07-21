"""No-step Stage 21A gradient compatibility audit on true source batches."""

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.cf_compat_kd_utils import gated_kd_loss
from trains.singleTask.missing_utils import MissingModalityWrapper, mode_to_mask
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed

from scripts.mosei.stage21a_common import (
    MISSING_MODES,
    atomic_frame,
    atomic_json,
    canonical_ids,
    ordered_id_sha,
    parse_sample_id,
    require_allowed_split,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--uniform-checkpoint", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--teacher-cache-manifest", required=True)
    parser.add_argument("--compatibility-cache", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=20)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def build_args(cli):
    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = 1111
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def gradients(loss, parameters, retain_graph):
    return torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, create_graph=False, allow_unused=True
    )


def gradient_stats(left, right, selection):
    dot = left_sq = right_sq = 0.0
    for index in selection:
        a, b = left[index], right[index]
        if a is not None:
            left_sq += float(a.detach().float().square().sum().cpu())
        if b is not None:
            right_sq += float(b.detach().float().square().sum().cpu())
        if a is not None and b is not None:
            dot += float((a.detach().float() * b.detach().float()).sum().cpu())
    left_norm, right_norm = math.sqrt(left_sq), math.sqrt(right_sq)
    denominator = left_norm * right_norm
    return {
        "cosine": float(dot / denominator if denominator else 0.0),
        "relative_gradient_norm": left_norm,
        "reference_gradient_norm": right_norm,
        "norm_ratio": float(left_norm / right_norm if right_norm else 0.0),
    }


def summarize(rows):
    result = {}
    for comparison in sorted(set(row["comparison"] for row in rows)):
        result[comparison] = {}
        for parameter_set in sorted(set(row["parameter_set"] for row in rows)):
            values = np.asarray(
                [row["cosine"] for row in rows if row["comparison"] == comparison and row["parameter_set"] == parameter_set],
                dtype=np.float64,
            )
            ratios = np.asarray(
                [row["norm_ratio"] for row in rows if row["comparison"] == comparison and row["parameter_set"] == parameter_set],
                dtype=np.float64,
            )
            result[comparison][parameter_set] = {
                "mean_cosine": float(values.mean()),
                "median_cosine": float(np.median(values)),
                "p25_cosine": float(np.quantile(values, 0.25)),
                "p75_cosine": float(np.quantile(values, 0.75)),
                "negative_ratio": float(np.mean(values < 0)),
                "mean_norm_ratio": float(ratios.mean()),
                "median_norm_ratio": float(np.median(ratios)),
                "batch_count": int(len(values)),
            }
    return result


def main():
    cli = parse_args()
    if cli.max_batches < 1 or cli.max_batches > 20:
        raise ValueError("Stage 21A permits at most 20 gradient batches.")
    setup_seed(2121)
    args = build_args(cli)
    require_allowed_split("train")
    dataset = MMDataset(args, mode="train")
    dataset_ids = canonical_ids(dataset.ids)
    dataset_videos = [parse_sample_id(sample_id)[0] for sample_id in dataset_ids]
    teacher_manifest = json.loads(Path(cli.teacher_cache_manifest).read_text())
    teacher_entry = teacher_manifest["entries"]["train"]
    if ordered_id_sha(dataset_ids) != teacher_entry["ordered_sample_id_sha256"]:
        raise RuntimeError("Teacher cache/dataset order mismatch.")
    if sha256_file(cli.teacher_cache) != teacher_entry["sha256"]:
        raise RuntimeError("Teacher cache SHA mismatch.")
    with np.load(cli.teacher_cache, allow_pickle=False) as archive:
        teacher_ids = archive["sample_id"].astype(str).tolist()
        teacher_prediction = archive["teacher_lav"].astype(np.float32)
    if teacher_ids != dataset_ids:
        raise RuntimeError("Teacher cache IDs differ from dataset IDs.")
    teacher_by_id = dict(zip(teacher_ids, teacher_prediction.tolist()))
    compatibility = pd.read_csv(cli.compatibility_cache)
    if compatibility["sample_id"].astype(str).tolist() != dataset_ids:
        raise RuntimeError("Compatibility cache IDs differ from dataset IDs.")
    compat_by_id = compatibility.set_index("sample_id").to_dict("index")
    pair_frame = pd.read_csv(cli.pair_manifest, sep="\t")
    pair_by_video = {
        key: value.to_dict("records")
        for key, value in pair_frame.groupby("video_id", sort=True)
    }
    args.seq_lens = dataset.get_seq_len()
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    model.load_state_dict(torch.load(cli.uniform_checkpoint, map_location=args.device), strict=True)
    model.eval()
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [value for _, value in named]
    selections = {
        "final_predictor": [
            index for index, (name, _) in enumerate(named)
            if name.startswith(("backbone.proj1.", "backbone.proj2.", "backbone.out_layer."))
        ],
        "final_fusion_projection": [
            index for index, (name, _) in enumerate(named) if name.startswith("backbone.projector_")
        ],
        "shared_encoder": [
            index for index, (name, _) in enumerate(named)
            if name.startswith(("backbone.encoder_c.", "backbone.align_c_", "backbone.self_attentions_c_"))
        ],
        "all_trainable_parameters": list(range(len(parameters))),
    }
    if any(not values for values in selections.values()):
        raise RuntimeError("At least one gradient parameter set is empty.")
    rows = []
    eligible = [video_id for video_id in sorted(pair_by_video) if len(pair_by_video[video_id])]
    for batch_number, video_id in enumerate(eligible[: cli.max_batches]):
        global_indices = sorted(set(
            [int(row["left_index"]) for row in pair_by_video[video_id]]
            + [int(row["right_index"]) for row in pair_by_video[video_id]]
        ))[:16]
        if len(global_indices) < 2:
            continue
        batch = default_collate([dataset[index] for index in global_indices])
        sample_ids = canonical_ids(batch["id"])
        mode = MISSING_MODES[batch_number % len(MISSING_MODES)]
        text = batch["text"].to(args.device)
        audio = batch["audio"].to(args.device)
        vision = batch["vision"].to(args.device)
        labels = batch["labels"]["M"].to(args.device).view(-1)
        mask = mode_to_mask(mode, len(sample_ids), args.device, audio.dtype)
        prediction = model(text, audio, vision, mask)["output_logit"].view(-1)
        supervised = F.mse_loss(prediction, labels)
        teacher = torch.as_tensor(
            [teacher_by_id[sample_id] for sample_id in sample_ids],
            device=args.device,
            dtype=prediction.dtype,
        )
        uniform = F.smooth_l1_loss(prediction, teacher)
        gate = torch.as_tensor(
            [compat_by_id[sample_id]["compat_{}".format(mode)] for sample_id in sample_ids],
            device=args.device,
            dtype=prediction.dtype,
        )
        cfcompat, _ = gated_kd_loss(prediction, teacher, gate)
        local = {global_index: position for position, global_index in enumerate(global_indices)}
        pair_losses = []
        for row in pair_by_video[video_id]:
            left, right = int(row["left_index"]), int(row["right_index"])
            if left in local and right in local:
                predicted_difference = prediction[local[left]] - prediction[local[right]]
                target_difference = labels[local[left]] - labels[local[right]]
                pair_losses.append((predicted_difference - target_difference).square())
        if not pair_losses:
            continue
        relative = torch.stack(pair_losses).mean()
        gradient_map = {
            "relative": gradients(relative, parameters, retain_graph=True),
            "supervised": gradients(supervised, parameters, retain_graph=True),
            "uniform_kd": gradients(uniform, parameters, retain_graph=True),
            "cfcompat_kd": gradients(cfcompat, parameters, retain_graph=False),
        }
        for comparison, reference in (
            ("relative_vs_supervised", "supervised"),
            ("relative_vs_uniform_kd", "uniform_kd"),
            ("relative_vs_cfcompat_kd", "cfcompat_kd"),
        ):
            for parameter_set, selection in selections.items():
                values = gradient_stats(gradient_map["relative"], gradient_map[reference], selection)
                rows.append(
                    {
                        "batch": batch_number,
                        "video_id": video_id,
                        "mode": mode,
                        "sample_count": len(sample_ids),
                        "relative_pair_count": len(pair_losses),
                        "comparison": comparison,
                        "parameter_set": parameter_set,
                        **values,
                    }
                )
    if len(set(row["batch"] for row in rows)) != cli.max_batches:
        raise RuntimeError("Gradient audit did not produce the requested number of batches.")
    summary = summarize(rows)
    cfcompat_all = summary["relative_vs_cfcompat_kd"]["all_trainable_parameters"]
    uniform_all = summary["relative_vs_uniform_kd"]["all_trainable_parameters"]
    result = {
        "created_at": utc_now(),
        "code_commit": git_head(),
        "optimizer_steps": 0,
        "batch_count": cli.max_batches,
        "parameter_sets": {key: len(value) for key, value in selections.items()},
        "summary": summary,
        "uniform_kd_conflict": {
            "median_cosine": uniform_all["median_cosine"],
            "negative_ratio": uniform_all["negative_ratio"],
            "risk": "HIGH" if uniform_all["median_cosine"] < -0.20 and uniform_all["negative_ratio"] > 0.70 else "NOT_HIGH",
        },
        "cfcompat_kd_conflict": {
            "median_cosine": cfcompat_all["median_cosine"],
            "negative_ratio": cfcompat_all["negative_ratio"],
            "risk": "HIGH" if cfcompat_all["median_cosine"] < -0.20 and cfcompat_all["negative_ratio"] > 0.70 else "NOT_HIGH",
        },
        "interpretation_limit": "Local gradient alignment does not prove training synergy.",
        "locked_test_access_count": 0,
    }
    output = Path(cli.output_root)
    atomic_frame(output / "gradients/gradient_compatibility.tsv", rows)
    atomic_json(output / "gradients/gradient_compatibility.json", result)
    print(json.dumps({"uniform": result["uniform_kd_conflict"], "cfcompat": result["cfcompat_kd_conflict"]}, indent=2))


if __name__ == "__main__":
    main()
