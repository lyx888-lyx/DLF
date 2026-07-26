#!/usr/bin/env python
"""Train frozen-protocol Stage23C Student screens and final cross-fit models."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stage23a_common import (
    MODES,
    atomic_json,
    build_subset_loader,
    ordered_id_sha,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.mgrd_utils import VectorizedHingeLoss
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
    regression_metrics,
    sample_missing_masks,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


OUT = ROOT / "result" / "oracle_distillation_v1" / "mosei"
V2 = ROOT / "result" / "arbiter_audit_v2" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23c"
SEED = 1111
SHUFFLE_SEEDS = (23611, 23612, 23613)
HIER_RATIOS = (0.25, 0.50, 1.00)
METHODS = (
    "S0_supervised_moddrop",
    "S1_uniform_single_teacher_kd",
    "S2_equal_ensemble_final_kd",
    "S3_fixed_stacking_final_kd",
    "S4_soft_oracle_final_kd",
    "S5_soft_oracle_hierarchical_kd",
    "S6_shuffled_soft_oracle_hierarchical_kd",
)
HEADS = (
    "logits_c",
    "logits_l_hetero",
    "logits_a_hetero",
    "logits_v_hetero",
)
MODE_HEADS = {
    "LAV": HEADS,
    "LA": ("logits_c", "logits_l_hetero", "logits_a_hetero"),
    "LV": ("logits_c", "logits_l_hetero", "logits_v_hetero"),
    "L": ("logits_c", "logits_l_hetero"),
}
CRITERION = nn.L1Loss()
COSINE = nn.CosineEmbeddingLoss()
HINGE = VectorizedHingeLoss()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "clean-screen",
            "student-screen",
            "freeze-pre-s6",
            "freeze-selection",
            "clean-final",
            "student-final",
        ),
    )
    parser.add_argument("--direction", choices=("A", "B"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--hier-ratio", type=float, choices=HIER_RATIOS)
    parser.add_argument("--shuffle-seed", type=int, choices=SHUFFLE_SEEDS)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_args(cli):
    args = get_config_regression(
        "DLF", "mosei", str(ROOT / "config" / "config.json")
    )
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = SEED
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def atomic_gzip_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as stream:
            frame.to_csv(stream, index=False, float_format="%.10g")
    os.replace(str(temporary), str(path))


def atomic_tsv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    os.replace(str(temporary), str(path))


def save_state(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(str(temporary), str(path))


def load_state(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    return model


def clear_cuda(*models):
    for model in models:
        if model is not None:
            model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def iter_limited(loader, limit):
    for step, batch in enumerate(loader, 1):
        if limit and step > limit:
            break
        yield step, batch


def mode_names(mask):
    mapping = {(1, 1, 0): "LA", (1, 0, 1): "LV", (1, 0, 0): "L"}
    return [mapping[tuple(row)] for row in mask.detach().cpu().long().tolist()]


def full_supervised_loss(model, batch, device, missing_generator):
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    labels = batch["labels"]["M"].to(device).view(-1, 1)
    full_mask = mode_to_mask("LAV", labels.size(0), device, audio.dtype)
    full, _ = compute_full_dlf_loss(
        model(text, audio, vision, full_mask),
        labels,
        CRITERION,
        COSINE,
        HINGE,
    )
    missing_mask = sample_missing_masks(
        labels.size(0), missing_generator, device, audio.dtype
    )
    missing_output = model(text, audio, vision, missing_mask)
    missing, _ = compute_task_loss(missing_output, labels, CRITERION)
    return (
        full + missing,
        missing_output,
        text,
        audio,
        vision,
        labels,
        missing_mask,
    )


def clean_supervised_loss(model, batch, device):
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    labels = batch["labels"]["M"].to(device).view(-1, 1)
    loss, _ = compute_full_dlf_loss(
        model(text, audio, vision),
        labels,
        CRITERION,
        COSINE,
        HINGE,
    )
    return loss


def evaluate_clean(model, loader, device, limit):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for _, batch in iter_limited(loader, limit):
            prediction = model(
                batch["text"].to(device),
                batch["audio"].to(device),
                batch["vision"].to(device),
            )["output_logit"]
            predictions.append(prediction.detach().cpu())
            labels.append(batch["labels"]["M"].view(-1, 1))
    values = regression_metrics(torch.cat(predictions), torch.cat(labels))
    values["Loss"] = values["MAE"]
    return {"LAV": values}


def evaluate_missing(model, loader, device, limit):
    model.eval()
    collected = {
        mode: {"prediction": [], "label": []} for mode in MODES
    }
    with torch.no_grad():
        for _, batch in iter_limited(loader, limit):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["labels"]["M"].view(-1, 1)
            for mode in MODES:
                mask = mode_to_mask(mode, label.size(0), device, audio.dtype)
                prediction = model(text, audio, vision, mask)[
                    "output_logit"
                ]
                collected[mode]["prediction"].append(
                    prediction.detach().cpu()
                )
                collected[mode]["label"].append(label)
    result = {}
    for mode, values in collected.items():
        result[mode] = regression_metrics(
            torch.cat(values["prediction"]), torch.cat(values["label"])
        )
        result[mode]["Loss"] = result[mode]["MAE"]
    return result


def role_sidecars(direction):
    root = V2 / "features" / "meta57" / f"direction_{direction}"
    output = {}
    for role in ("inner_train", "inner_valid", "outer_evaluation"):
        frame = pd.read_csv(
            root / f"{role}_targets_and_audit.csv",
            usecols=[
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "role",
            ],
            dtype={"sample_id": str, "video_id": str},
        )
        per_sample = frame[
            ["sample_id", "video_id", "train_index"]
        ].drop_duplicates()
        if len(frame) != len(per_sample) * len(MODES):
            raise RuntimeError(f"{direction} {role} missing mode rows")
        if per_sample["train_index"].duplicated().any():
            raise RuntimeError(f"{direction} {role} duplicate train index")
        output[role] = (frame, per_sample)
    sources = {
        role: set(values[1]["video_id"]) for role, values in output.items()
    }
    if (
        sources["inner_train"] & sources["inner_valid"]
        or sources["inner_train"] & sources["outer_evaluation"]
        or sources["inner_valid"] & sources["outer_evaluation"]
    ):
        raise RuntimeError("Source leakage in role sidecars")
    return output


def configure_dataset(cli):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    setup_seed(SEED)
    args = build_args(cli)
    dataset = MMDataset(args, mode="train")
    args.seq_lens = dataset.get_seq_len()
    return args, dataset


def loader(dataset, indices, args, cli, shuffle):
    return build_subset_loader(
        dataset,
        indices,
        args.batch_size,
        shuffle,
        SEED,
        cli.num_workers,
    )


class TargetBank:
    def __init__(self, direction):
        self.path = OUT / "targets" / f"direction_{direction}_training_targets.npz"
        self.sha256 = sha256_file(self.path)
        values = np.load(self.path)
        self.values = {key: values[key] for key in values.files}
        self.index = {
            (int(train_index), str(mode)): row
            for row, (train_index, mode) in enumerate(
                zip(self.values["train_index"], self.values["mode"])
            )
        }
        if len(self.index) != len(self.values["train_index"]):
            raise RuntimeError("Duplicate target bank key")

    def rows(self, train_indices, modes):
        return np.asarray(
            [
                self.index[(int(train_index), str(mode))]
                for train_index, mode in zip(train_indices, modes)
            ],
            dtype=np.int64,
        )

    def tensor(self, column, rows, device, dtype):
        return torch.as_tensor(
            self.values[column][rows],
            device=device,
            dtype=dtype,
        ).view(-1)


def method_run_id(method, hier_ratio=None, shuffle_seed=None):
    if method == "S5_soft_oracle_hierarchical_kd":
        if hier_ratio is None:
            raise ValueError("S5 requires a hierarchical ratio")
        return f"S5_hier_ratio{str(hier_ratio).replace('.', 'p')}"
    if method == "S6_shuffled_soft_oracle_hierarchical_kd":
        if shuffle_seed is None:
            raise ValueError("S6 requires a shuffle seed")
        return f"S6_shuffle_seed{shuffle_seed}"
    return method.split("_", 1)[0]


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def train_screen(
    name,
    model,
    train_loader,
    valid_loader,
    args,
    cli,
    step_function,
    evaluator,
    output_dir,
    metadata,
):
    checkpoint = output_dir / "best_inner_valid.pth"
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and checkpoint.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "COMPLETED"
            and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
        ):
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return manifest

    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(args.patience)
    )
    best = float("inf")
    best_epoch = 0
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, int(cli.max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        totals = {
            "loss": 0.0,
            "supervised": 0.0,
            "final_kd": 0.0,
            "hier_kd": 0.0,
        }
        batches = 0
        for step, batch in iter_limited(train_loader, cli.smoke_batches):
            loss, details = step_function(model, batch, epoch, step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} non-finite loss")
            loss.backward()
            batches += 1
            totals["loss"] += float(loss.detach().cpu())
            for key in ("supervised", "final_kd", "hier_kd"):
                totals[key] += float(details.get(key, 0.0))
            should_step = (
                step % int(args.update_epochs) == 0
                or (cli.smoke_batches and step == cli.smoke_batches)
                or (not cli.smoke_batches and step == len(train_loader))
            )
            if should_step:
                nn.utils.clip_grad_value_(
                    model.parameters(), float(args.grad_clip)
                )
                optimizer.step()
                optimizer.zero_grad()
        if not batches:
            raise RuntimeError(f"{name} empty train loader")
        metrics = evaluator(
            model, valid_loader, args.device, cli.smoke_batches
        )
        score = (
            float(metrics["LAV"]["MAE"])
            if set(metrics) == {"LAV"}
            else float(validation_objective(metrics))
        )
        if not math.isfinite(score):
            raise FloatingPointError(f"{name} non-finite validation score")
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / batches,
            "train_supervised_loss": totals["supervised"] / batches,
            "train_final_kd_loss": totals["final_kd"] / batches,
            "train_hier_kd_loss": totals["hier_kd"] / batches,
            "inner_valid_J": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "wall_seconds": time.perf_counter() - started,
        }
        for mode, values in metrics.items():
            for metric, value in values.items():
                row[f"{mode}_{metric}"] = float(value)
        history.append(row)
        atomic_tsv(pd.DataFrame(history), output_dir / "epoch_metrics.tsv")
        print(
            f"stage23c {name} epoch={epoch} inner_J={score:.6f} "
            f"best={min(best, score):.6f}",
            flush=True,
        )
        if score <= best - 1e-6:
            best = score
            best_epoch = epoch
            save_state(model, checkpoint)
        if not cli.smoke_batches and epoch - best_epoch >= int(cli.patience):
            break
        if cli.smoke_batches:
            break
    manifest = {
        "stage": "Stage23C inner-valid Student screen",
        "status": "COMPLETED",
        "run_id": name,
        "student_seed": SEED,
        "best_epoch": int(best_epoch),
        "best_inner_valid_J": float(best),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": parameter_count(model),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "outer_evaluation_access_count": 0,
        "completed_at": utc_now(),
        **metadata,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def clean_screen(cli):
    args, dataset = configure_dataset(cli)
    roles = role_sidecars(cli.direction)
    train_indices = (
        roles["inner_train"][1]["train_index"].astype(int).tolist()
    )
    valid_indices = (
        roles["inner_valid"][1]["train_index"].astype(int).tolist()
    )
    setup_seed(SEED)
    model = DLF(args).to(args.device)
    train_loader = loader(dataset, train_indices, args, cli, True)
    valid_loader = loader(dataset, valid_indices, args, cli, False)

    def step(current, batch, epoch, batch_step):
        del epoch, batch_step
        loss = clean_supervised_loss(current, batch, args.device)
        return loss, {"supervised": float(loss.detach().cpu())}

    group = "smoke" if cli.smoke_batches else "training"
    output = OUT / group / f"direction_{cli.direction}" / "screen" / "clean"
    manifest = train_screen(
        f"direction_{cli.direction}_clean",
        model,
        train_loader,
        valid_loader,
        args,
        cli,
        step,
        evaluate_clean,
        output,
        {
            "direction": cli.direction,
            "method": "clean_initialization_component",
            "training_samples": len(train_indices),
            "inner_valid_samples": len(valid_indices),
            "training_sample_sha256": ordered_id_sha(
                roles["inner_train"][1]["sample_id"].tolist()
            ),
        },
    )
    clear_cuda(model)
    return manifest


def kd_target_columns(method, scope, shuffle_seed):
    if method == "S2_equal_ensemble_final_kd":
        return "equal_final", None
    if method == "S3_fixed_stacking_final_kd":
        return (
            "screen_fixed_final"
            if scope == "screen"
            else "final_fixed_final"
        ), None
    if method == "S4_soft_oracle_final_kd":
        return "oracle__output_logit", None
    if method == "S5_soft_oracle_hierarchical_kd":
        return "oracle__output_logit", "oracle__"
    if method == "S6_shuffled_soft_oracle_hierarchical_kd":
        if shuffle_seed is None:
            raise ValueError("S6 requires shuffle seed")
        prefix = f"s6_{scope}_seed{shuffle_seed}__"
        return prefix + "output_logit", prefix
    return None, None


def student_step_function(
    method,
    args,
    missing_generator,
    target_bank,
    scope,
    hier_ratio,
    shuffle_seed,
    uniform_teacher,
):
    final_column, hier_prefix = kd_target_columns(
        method, scope, shuffle_seed
    )

    def step(current, batch, epoch, batch_step):
        del epoch, batch_step
        (
            supervised,
            missing_output,
            text,
            audio,
            vision,
            labels,
            missing_mask,
        ) = full_supervised_loss(
            current, batch, args.device, missing_generator
        )
        final_kd = labels.new_tensor(0.0)
        hier_kd = labels.new_tensor(0.0)
        if method == "S1_uniform_single_teacher_kd":
            with torch.no_grad():
                target = uniform_teacher(text, audio, vision)[
                    "output_logit"
                ].view(-1)
            final_kd = F.smooth_l1_loss(
                missing_output["output_logit"].view(-1), target
            )
        elif final_column is not None:
            names = mode_names(missing_mask)
            rows = target_bank.rows(
                batch["index"].view(-1).tolist(), names
            )
            target = target_bank.tensor(
                final_column,
                rows,
                args.device,
                labels.dtype,
            )
            final_kd = F.smooth_l1_loss(
                missing_output["output_logit"].view(-1), target
            )
            if hier_prefix is not None:
                per_sample = labels.new_zeros(labels.size(0))
                active_count = labels.new_zeros(labels.size(0))
                for head in HEADS:
                    active = torch.as_tensor(
                        [head in MODE_HEADS[mode] for mode in names],
                        device=args.device,
                        dtype=labels.dtype,
                    )
                    head_target = target_bank.tensor(
                        hier_prefix + head,
                        rows,
                        args.device,
                        labels.dtype,
                    )
                    head_loss = F.smooth_l1_loss(
                        missing_output[head].view(-1),
                        head_target,
                        reduction="none",
                    )
                    per_sample += active * head_loss
                    active_count += active
                if torch.any(active_count <= 0):
                    raise RuntimeError("No active hierarchical head")
                hier_kd = torch.mean(per_sample / active_count)
        loss = supervised + final_kd
        if hier_prefix is not None:
            loss = loss + float(hier_ratio) * hier_kd
        return loss, {
            "supervised": float(supervised.detach().cpu()),
            "final_kd": float(final_kd.detach().cpu()),
            "hier_kd": float(hier_kd.detach().cpu()),
        }

    return step


def read_pre_s6_selection():
    path = OUT / "protocol" / "frozen_pre_s6_selection.json"
    if not path.exists():
        raise RuntimeError("Freeze S5 hierarchical ratio before S6")
    return json.loads(path.read_text(encoding="utf-8"))


def student_screen(cli):
    if cli.method is None:
        raise ValueError("--method is required")
    if cli.method == "S5_soft_oracle_hierarchical_kd" and cli.hier_ratio is None:
        raise ValueError("S5 screen requires --hier-ratio")
    if (
        cli.method == "S6_shuffled_soft_oracle_hierarchical_kd"
        and cli.shuffle_seed is None
    ):
        raise ValueError("S6 screen requires --shuffle-seed")
    if cli.method == "S6_shuffled_soft_oracle_hierarchical_kd":
        selected = read_pre_s6_selection()
        frozen_ratio = float(
            selected["directions"][cli.direction]["selected_hier_ratio"]
        )
        if cli.hier_ratio is not None and cli.hier_ratio != frozen_ratio:
            raise RuntimeError("S6 hierarchical ratio differs from frozen S5")
        cli.hier_ratio = frozen_ratio

    args, dataset = configure_dataset(cli)
    roles = role_sidecars(cli.direction)
    train_indices = (
        roles["inner_train"][1]["train_index"].astype(int).tolist()
    )
    valid_indices = (
        roles["inner_valid"][1]["train_index"].astype(int).tolist()
    )
    group = "smoke" if cli.smoke_batches else "training"
    clean_dir = (
        OUT / group / f"direction_{cli.direction}" / "screen" / "clean"
    )
    clean_manifest = json.loads(
        (clean_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    clean_checkpoint = Path(clean_manifest["checkpoint"])
    if sha256_file(clean_checkpoint) != clean_manifest["checkpoint_sha256"]:
        raise RuntimeError("Clean screen checkpoint SHA mismatch")

    setup_seed(SEED)
    backbone = load_state(DLF(args), clean_checkpoint, "cpu")
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    initial_parameter_count = parameter_count(model)
    uniform_teacher = None
    if cli.method == "S1_uniform_single_teacher_kd":
        uniform_teacher = load_state(
            DLF(args), clean_checkpoint, "cpu"
        ).to(args.device)
        uniform_teacher.eval()
        for parameter in uniform_teacher.parameters():
            parameter.requires_grad_(False)
    target_bank = (
        None
        if cli.method
        in ("S0_supervised_moddrop", "S1_uniform_single_teacher_kd")
        else TargetBank(cli.direction)
    )
    missing_generator = torch.Generator().manual_seed(SEED + 104729)
    train_loader = loader(dataset, train_indices, args, cli, True)
    valid_loader = loader(dataset, valid_indices, args, cli, False)
    step = student_step_function(
        cli.method,
        args,
        missing_generator,
        target_bank,
        "screen",
        cli.hier_ratio,
        cli.shuffle_seed,
        uniform_teacher,
    )
    run_id = method_run_id(
        cli.method, cli.hier_ratio, cli.shuffle_seed
    )
    output = (
        OUT
        / group
        / f"direction_{cli.direction}"
        / "screen"
        / run_id
    )
    manifest = train_screen(
        f"direction_{cli.direction}_{run_id}",
        model,
        train_loader,
        valid_loader,
        args,
        cli,
        step,
        evaluate_missing,
        output,
        {
            "direction": cli.direction,
            "method": cli.method,
            "hier_ratio": cli.hier_ratio,
            "shuffle_seed": cli.shuffle_seed,
            "lambda_final": 1.0,
            "initialization_checkpoint": str(clean_checkpoint.resolve()),
            "initialization_checkpoint_sha256": sha256_file(
                clean_checkpoint
            ),
            "training_target_sha256": (
                target_bank.sha256 if target_bank is not None else None
            ),
            "training_samples": len(train_indices),
            "inner_valid_samples": len(valid_indices),
            "training_sample_sha256": ordered_id_sha(
                roles["inner_train"][1]["sample_id"].tolist()
            ),
            "same_parameter_count_as_protocol": (
                initial_parameter_count == parameter_count(model)
            ),
            "student_inference_inputs": "text,audio,vision,mode mask only",
        },
    )
    clear_cuda(model, uniform_teacher)
    return manifest


def screen_manifest(direction, run_id):
    path = (
        OUT
        / "training"
        / f"direction_{direction}"
        / "screen"
        / run_id
        / "run_manifest.json"
    )
    if not path.exists():
        raise RuntimeError(f"Missing screen manifest {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["status"] != "COMPLETED":
        raise RuntimeError(f"Incomplete screen {path}")
    return result, path


def freeze_pre_s6():
    output = {
        "stage": "Stage23C pre-S6 frozen selection",
        "selection_scope": "inner-valid only",
        "outer_evaluation_access_count": 0,
        "directions": {},
        "frozen_at": utc_now(),
    }
    for direction in ("A", "B"):
        candidates = []
        for ratio in HIER_RATIOS:
            run_id = method_run_id(
                "S5_soft_oracle_hierarchical_kd", ratio, None
            )
            manifest, path = screen_manifest(direction, run_id)
            candidates.append(
                {
                    "hier_ratio": ratio,
                    "best_epoch": manifest["best_epoch"],
                    "inner_valid_J": manifest["best_inner_valid_J"],
                    "manifest_path": str(path.resolve()),
                    "manifest_sha256": sha256_file(path),
                }
            )
        selected = sorted(
            candidates, key=lambda row: (row["inner_valid_J"], row["hier_ratio"])
        )[0]
        output["directions"][direction] = {
            "selected_hier_ratio": selected["hier_ratio"],
            "selected_s5_epoch": selected["best_epoch"],
            "selected_inner_valid_J": selected["inner_valid_J"],
            "candidates": candidates,
        }
    path = OUT / "protocol" / "frozen_pre_s6_selection.json"
    atomic_json(path, output)
    print(json.dumps(output, indent=2, sort_keys=True))


def freeze_selection():
    pre = read_pre_s6_selection()
    output = {
        "stage": "Stage23C frozen Student selection",
        "selection_scope": "inner-valid only",
        "outer_evaluation_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "directions": {},
        "frozen_at": utc_now(),
    }
    parameter_counts = []
    for direction in ("A", "B"):
        methods = {}
        for method in METHODS[:5]:
            run_id = method_run_id(method)
            manifest, path = screen_manifest(direction, run_id)
            methods[method] = {
                "run_id": run_id,
                "best_epoch": manifest["best_epoch"],
                "inner_valid_J": manifest["best_inner_valid_J"],
                "hier_ratio": None,
                "shuffle_seed": None,
                "manifest_path": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
            }
            parameter_counts.append(manifest["parameter_count"])
        ratio = float(
            pre["directions"][direction]["selected_hier_ratio"]
        )
        run_id = method_run_id(
            "S5_soft_oracle_hierarchical_kd", ratio, None
        )
        manifest, path = screen_manifest(direction, run_id)
        methods["S5_soft_oracle_hierarchical_kd"] = {
            "run_id": run_id,
            "best_epoch": manifest["best_epoch"],
            "inner_valid_J": manifest["best_inner_valid_J"],
            "hier_ratio": ratio,
            "shuffle_seed": None,
            "manifest_path": str(path.resolve()),
            "manifest_sha256": sha256_file(path),
        }
        parameter_counts.append(manifest["parameter_count"])
        shuffled = {}
        for seed in SHUFFLE_SEEDS:
            run_id = method_run_id(
                "S6_shuffled_soft_oracle_hierarchical_kd",
                ratio,
                seed,
            )
            manifest, path = screen_manifest(direction, run_id)
            shuffled[str(seed)] = {
                "run_id": run_id,
                "best_epoch": manifest["best_epoch"],
                "inner_valid_J": manifest["best_inner_valid_J"],
                "hier_ratio": ratio,
                "shuffle_seed": seed,
                "manifest_path": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
            }
            parameter_counts.append(manifest["parameter_count"])
        methods[
            "S6_shuffled_soft_oracle_hierarchical_kd"
        ] = shuffled
        output["directions"][direction] = {
            "methods": methods,
            "selected_hier_ratio": ratio,
        }
    if len(set(parameter_counts)) != 1:
        raise RuntimeError("Student parameter counts differ")
    output["student_parameter_count"] = parameter_counts[0]
    path = OUT / "protocol" / "frozen_student_selection.json"
    atomic_json(path, output)
    state_path = RUNTIME / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "STUDENT_SELECTION_FROZEN_OUTER_NOT_ACCESSED",
            "student_selection_path": str(path.resolve()),
            "student_selection_sha256": sha256_file(path),
            "outer_evaluation_access_count": 0,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "updated_at": utc_now(),
        }
    )
    atomic_json(state_path, state)
    print(json.dumps(output, indent=2, sort_keys=True))


def train_fixed_epochs(
    name,
    model,
    train_loader,
    args,
    cli,
    epochs,
    step_function,
    output_dir,
    metadata,
    learning_rate_schedule=None,
):
    checkpoint = output_dir / "final_development_checkpoint.pth"
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists() and checkpoint.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "COMPLETED"
            and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
        ):
            load_state(model, checkpoint, args.device)
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return manifest
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate))
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        if learning_rate_schedule is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate_schedule[epoch - 1])
        model.train()
        optimizer.zero_grad()
        totals = {
            "loss": 0.0,
            "supervised": 0.0,
            "final_kd": 0.0,
            "hier_kd": 0.0,
        }
        batches = 0
        for step, batch in enumerate(train_loader, 1):
            loss, details = step_function(model, batch, epoch, step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name} non-finite loss")
            loss.backward()
            batches += 1
            totals["loss"] += float(loss.detach().cpu())
            for key in ("supervised", "final_kd", "hier_kd"):
                totals[key] += float(details.get(key, 0.0))
            if step % int(args.update_epochs) == 0 or step == len(train_loader):
                nn.utils.clip_grad_value_(
                    model.parameters(), float(args.grad_clip)
                )
                optimizer.step()
                optimizer.zero_grad()
        history.append(
            {
                "epoch": epoch,
                **{
                    f"train_{key}": value / batches
                    for key, value in totals.items()
                },
                "wall_seconds": time.perf_counter() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        atomic_tsv(
            pd.DataFrame(history), output_dir / "retrain_epoch_metrics.tsv"
        )
        print(
            f"stage23c {name} final epoch={epoch}/{epochs} "
            f"loss={totals['loss']/batches:.6f}",
            flush=True,
        )
    save_state(model, checkpoint)
    manifest = {
        "stage": "Stage23C full development-side fixed-epoch retraining",
        "status": "COMPLETED",
        "run_id": name,
        "student_seed": SEED,
        "frozen_epochs": int(epochs),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameter_count": parameter_count(model),
        "outer_prediction_written": False,
        "outer_label_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "completed_at": utc_now(),
        "frozen_learning_rate_schedule": learning_rate_schedule,
        **metadata,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def clean_final(cli):
    args, dataset = configure_dataset(cli)
    roles = role_sidecars(cli.direction)
    development = pd.concat(
        [roles["inner_train"][1], roles["inner_valid"][1]],
        ignore_index=True,
    )
    clean_screen_path = (
        OUT
        / "training"
        / f"direction_{cli.direction}"
        / "screen"
        / "clean"
        / "run_manifest.json"
    )
    clean_screen_manifest = json.loads(
        clean_screen_path.read_text(encoding="utf-8")
    )
    epochs = int(clean_screen_manifest["best_epoch"])
    screen_history = pd.read_csv(
        clean_screen_path.parent / "epoch_metrics.tsv", sep="\t"
    )
    learning_rate_schedule = [float(args.learning_rate)] + screen_history[
        "learning_rate"
    ].astype(float).tolist()[: max(epochs - 1, 0)]
    setup_seed(SEED)
    model = DLF(args).to(args.device)
    train_loader = loader(
        dataset,
        development["train_index"].astype(int).tolist(),
        args,
        cli,
        True,
    )

    def step(current, batch, epoch, batch_step):
        del epoch, batch_step
        loss = clean_supervised_loss(current, batch, args.device)
        return loss, {"supervised": float(loss.detach().cpu())}

    output = (
        OUT
        / "training"
        / f"direction_{cli.direction}"
        / "final"
        / "clean"
    )
    manifest = train_fixed_epochs(
        f"direction_{cli.direction}_clean_final",
        model,
        train_loader,
        args,
        cli,
        epochs,
        step,
        output,
        {
            "direction": cli.direction,
            "method": "clean_initialization_component",
            "screen_manifest_path": str(clean_screen_path.resolve()),
            "screen_manifest_sha256": sha256_file(clean_screen_path),
            "training_samples": len(development),
            "training_sample_sha256": ordered_id_sha(
                development["sample_id"].tolist()
            ),
        },
        learning_rate_schedule,
    )
    clear_cuda(model)
    return manifest


def frozen_method_config(direction, method, shuffle_seed):
    path = OUT / "protocol" / "frozen_student_selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    value = selection["directions"][direction]["methods"][method]
    if method == "S6_shuffled_soft_oracle_hierarchical_kd":
        value = value[str(shuffle_seed)]
    return value, path


def predict_outer_label_free(
    model, dataset, indices, direction, method_id, args, cli
):
    outer_loader = loader(dataset, indices, args, cli, False)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in outer_loader:
            # Deliberately do not index batch["labels"] anywhere in this path.
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            identifiers = [str(value) for value in batch["id"]]
            train_indices = batch["index"].view(-1).tolist()
            for mode in MODES:
                mask = mode_to_mask(
                    mode, text.size(0), args.device, audio.dtype
                )
                prediction = (
                    model(text, audio, vision, mask)["output_logit"]
                    .view(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                for offset, sample_id in enumerate(identifiers):
                    rows.append(
                        {
                            "direction": direction,
                            "method": method_id,
                            "sample_id": sample_id,
                            "train_index": int(train_indices[offset]),
                            "mode": mode,
                            "prediction": float(prediction[offset]),
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.duplicated(["sample_id", "mode"]).any():
        raise RuntimeError("Duplicate outer Student prediction")
    if len(frame) != len(indices) * len(MODES):
        raise RuntimeError("Incomplete outer Student prediction")
    if "label" in frame.columns:
        raise RuntimeError("Label leaked into frozen prediction ledger")
    return frame


def student_final(cli):
    if cli.method is None:
        raise ValueError("--method is required")
    if (
        cli.method == "S6_shuffled_soft_oracle_hierarchical_kd"
        and cli.shuffle_seed is None
    ):
        raise ValueError("S6 final requires --shuffle-seed")
    config, selection_path = frozen_method_config(
        cli.direction, cli.method, cli.shuffle_seed
    )
    cli.hier_ratio = config["hier_ratio"]
    if cli.method == "S6_shuffled_soft_oracle_hierarchical_kd":
        cli.shuffle_seed = int(config["shuffle_seed"])
    args, dataset = configure_dataset(cli)
    roles = role_sidecars(cli.direction)
    development = pd.concat(
        [roles["inner_train"][1], roles["inner_valid"][1]],
        ignore_index=True,
    )
    outer_indices = (
        roles["outer_evaluation"][1]["train_index"].astype(int).tolist()
    )
    clean_dir = (
        OUT
        / "training"
        / f"direction_{cli.direction}"
        / "final"
        / "clean"
    )
    clean_manifest = json.loads(
        (clean_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    clean_checkpoint = Path(clean_manifest["checkpoint"])
    if sha256_file(clean_checkpoint) != clean_manifest["checkpoint_sha256"]:
        raise RuntimeError("Final clean checkpoint SHA mismatch")
    setup_seed(SEED)
    backbone = load_state(DLF(args), clean_checkpoint, "cpu")
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    uniform_teacher = None
    if cli.method == "S1_uniform_single_teacher_kd":
        uniform_teacher = load_state(
            DLF(args), clean_checkpoint, "cpu"
        ).to(args.device)
        uniform_teacher.eval()
        for parameter in uniform_teacher.parameters():
            parameter.requires_grad_(False)
    target_bank = (
        None
        if cli.method
        in ("S0_supervised_moddrop", "S1_uniform_single_teacher_kd")
        else TargetBank(cli.direction)
    )
    missing_generator = torch.Generator().manual_seed(SEED + 104729)
    train_loader = loader(
        dataset,
        development["train_index"].astype(int).tolist(),
        args,
        cli,
        True,
    )
    step = student_step_function(
        cli.method,
        args,
        missing_generator,
        target_bank,
        "final",
        cli.hier_ratio,
        cli.shuffle_seed,
        uniform_teacher,
    )
    run_id = method_run_id(
        cli.method, cli.hier_ratio, cli.shuffle_seed
    )
    output = (
        OUT
        / "training"
        / f"direction_{cli.direction}"
        / "final"
        / run_id
    )
    screen_manifest_path = Path(config["manifest_path"])
    screen_history = pd.read_csv(
        screen_manifest_path.parent / "epoch_metrics.tsv", sep="\t"
    )
    frozen_epochs = int(config["best_epoch"])
    learning_rate_schedule = [float(args.learning_rate)] + screen_history[
        "learning_rate"
    ].astype(float).tolist()[: max(frozen_epochs - 1, 0)]
    manifest = train_fixed_epochs(
        f"direction_{cli.direction}_{run_id}_final",
        model,
        train_loader,
        args,
        cli,
        frozen_epochs,
        step,
        output,
        {
            "direction": cli.direction,
            "method": cli.method,
            "hier_ratio": cli.hier_ratio,
            "shuffle_seed": cli.shuffle_seed,
            "lambda_final": 1.0,
            "selection_manifest_path": str(selection_path.resolve()),
            "selection_manifest_sha256": sha256_file(selection_path),
            "initialization_checkpoint": str(clean_checkpoint.resolve()),
            "initialization_checkpoint_sha256": sha256_file(
                clean_checkpoint
            ),
            "training_target_sha256": (
                target_bank.sha256 if target_bank is not None else None
            ),
            "training_samples": len(development),
            "training_sample_sha256": ordered_id_sha(
                development["sample_id"].tolist()
            ),
            "student_inference_inputs": "text,audio,vision,mode mask only",
        },
        learning_rate_schedule,
    )
    prediction_path = output / "outer_predictions_label_free.csv.gz"
    manifest_path = output / "run_manifest.json"
    if not prediction_path.exists():
        predictions = predict_outer_label_free(
            model,
            dataset,
            outer_indices,
            cli.direction,
            run_id,
            args,
            cli,
        )
        atomic_gzip_csv(predictions, prediction_path)
        manifest.update(
            {
                "outer_prediction_written": True,
                "outer_prediction_path": str(prediction_path.resolve()),
                "outer_prediction_sha256": sha256_file(prediction_path),
                "outer_prediction_rows": len(predictions),
                "outer_label_access_count": 0,
                "outer_prediction_pass_count": 1,
                "outer_labels_in_prediction_file": False,
            }
        )
        atomic_json(manifest_path, manifest)
    else:
        if manifest.get("outer_prediction_sha256") != sha256_file(
            prediction_path
        ):
            raise RuntimeError("Existing outer prediction SHA mismatch")
    clear_cuda(model, uniform_teacher)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main():
    cli = parse_args()
    protocol = json.loads(
        (
            OUT / "protocol" / "frozen_protocol_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if not protocol["student_screen_training_authorized"]:
        raise RuntimeError("Student screen training is not authorized")
    if (
        protocol["official_valid_authorized"]
        or protocol["test_authorized"]
        or protocol["judge_training_authorized"]
        or protocol["joint_specialized_moe_authorized"]
    ):
        raise RuntimeError("Forbidden Stage23C authorization is open")
    if cli.phase not in ("freeze-pre-s6", "freeze-selection"):
        if cli.direction is None or cli.gpu_id is None:
            raise ValueError("--direction and --gpu-id are required")
    if cli.phase == "clean-screen":
        clean_screen(cli)
    elif cli.phase == "student-screen":
        student_screen(cli)
    elif cli.phase == "freeze-pre-s6":
        freeze_pre_s6()
    elif cli.phase == "freeze-selection":
        freeze_selection()
    elif cli.phase == "clean-final":
        clean_final(cli)
    else:
        student_final(cli)


if __name__ == "__main__":
    main()
