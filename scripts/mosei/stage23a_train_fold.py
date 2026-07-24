"""Train one outer source-disjoint fold and emit Expert OOF predictions.

Only the MOSEI train split is constructed.  The outer holdout is never used
for checkpoint selection.  Each candidate is selected on a source-disjoint
inner subset of the outer-training sources.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from stage23a_common import (
    EXPERTS,
    MODES,
    MISSING_MODES,
    RESULT_ROOT,
    atomic_csv,
    atomic_json,
    build_subset_loader,
    git_head,
    load_preregistered,
    load_split_manifest,
    ordered_id_sha,
    sha256_file,
    source_video,
)

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_from_deltas,
    gated_kd_loss,
)
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


CRITERION = nn.L1Loss()
COSINE = nn.CosineEmbeddingLoss()
HINGE = VectorizedHingeLoss()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", required=True, type=int, choices=(0, 1))
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--config-file", default=str(ROOT / "config" / "config.json"))
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_args(cli):
    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def finite_metrics(metrics):
    return all(
        math.isfinite(float(value))
        for mode in metrics.values()
        for value in mode.values()
    )


def iter_limited(loader, limit):
    for index, batch in enumerate(loader, 1):
        if limit and index > limit:
            break
        yield index, batch


def evaluate_clean(model, loader, device, limit):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for _, batch in iter_limited(loader, limit):
            output = model(
                batch["text"].to(device),
                batch["audio"].to(device),
                batch["vision"].to(device),
            )["output_logit"]
            predictions.append(output.detach().cpu())
            labels.append(batch["labels"]["M"].view(-1, 1))
    values = regression_metrics(torch.cat(predictions), torch.cat(labels))
    values["Loss"] = values["MAE"]
    return {"LAV": values}


def evaluate_missing(model, loader, device, limit):
    model.eval()
    collected = {
        mode: {"prediction": [], "label": []}
        for mode in MODES
    }
    with torch.no_grad():
        for _, batch in iter_limited(loader, limit):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["labels"]["M"].view(-1, 1)
            for mode in MODES:
                mask = mode_to_mask(mode, label.size(0), device, audio.dtype)
                prediction = model(text, audio, vision, mask)["output_logit"]
                collected[mode]["prediction"].append(prediction.detach().cpu())
                collected[mode]["label"].append(label)
    result = {}
    for mode, values in collected.items():
        result[mode] = regression_metrics(
            torch.cat(values["prediction"]), torch.cat(values["label"])
        )
        result[mode]["Loss"] = result[mode]["MAE"]
    return result


def save_state(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


def load_state(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    return model


def train_with_inner_selection(
    name,
    model,
    train_loader,
    valid_loader,
    args,
    cli,
    train_step,
    evaluate,
    output_dir,
):
    checkpoint = output_dir / "best_inner_train_only.pth"
    completion = output_dir / "run_manifest.json"
    if completion.exists() and checkpoint.exists():
        manifest = json.loads(completion.read_text())
        if (
            manifest.get("status") == "COMPLETED"
            and manifest.get("checkpoint_sha256") == sha256_file(checkpoint)
        ):
            load_state(model, checkpoint, args.device)
            return model, manifest

    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate))
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(args.patience)
    )
    best, best_epoch = float("inf"), 0
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, int(cli.max_epochs) + 1):
        model.train()
        optimizer.zero_grad()
        total = 0.0
        batches = 0
        for step, batch in iter_limited(train_loader, cli.smoke_batches):
            loss, details = train_step(model, batch, epoch, step)
            if not torch.isfinite(loss):
                raise FloatingPointError("{} produced non-finite loss".format(name))
            loss.backward()
            batches += 1
            total += float(loss.detach().cpu())
            should_step = (
                step % int(args.update_epochs) == 0
                or (cli.smoke_batches and step == cli.smoke_batches)
                or (not cli.smoke_batches and step == len(train_loader))
            )
            if should_step:
                nn.utils.clip_grad_value_(model.parameters(), float(args.grad_clip))
                optimizer.step()
                optimizer.zero_grad()
        if not batches:
            raise RuntimeError("{} empty train loader".format(name))
        metrics = evaluate(model, valid_loader, args.device, cli.smoke_batches)
        if not finite_metrics(metrics):
            raise FloatingPointError("{} non-finite inner metrics".format(name))
        score = (
            float(metrics["LAV"]["MAE"])
            if set(metrics) == {"LAV"}
            else float(validation_objective(metrics))
        )
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_loss": total / batches,
            "inner_J": score,
            "wall_seconds": time.perf_counter() - started,
        }
        for mode, values in metrics.items():
            for key, value in values.items():
                row["{}_{}".format(mode, key)] = float(value)
        rows.append(row)
        atomic_csv(pd.DataFrame(rows), output_dir / "epoch_metrics.csv")
        print(
            "stage23a {} epoch={} inner_J={:.6f} best={:.6f}".format(
                name, epoch, score, min(best, score)
            ),
            flush=True,
        )
        if score <= best - 1e-6:
            best, best_epoch = score, epoch
            save_state(model, checkpoint)
        if not cli.smoke_batches and epoch - best_epoch >= int(cli.patience):
            break
        if cli.smoke_batches:
            break
    load_state(model, checkpoint, args.device)
    manifest = {
        "stage": "Stage23A outer-fold expert cross-fit",
        "expert_or_component": name,
        "outer_fold": int(cli.outer_fold),
        "status": "COMPLETED",
        "best_inner_train_only_J": float(best),
        "best_epoch": int(best_epoch),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "code_commit": git_head(),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "completed_at": utc_now(),
    }
    atomic_json(completion, manifest)
    return model, manifest


def full_supervised_loss(model, batch, device, missing_generator, include_missing):
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    labels = batch["labels"]["M"].to(device).view(-1, 1)
    if include_missing:
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
        missing, _ = compute_task_loss(
            model(text, audio, vision, missing_mask), labels, CRITERION
        )
        return full + missing, (text, audio, vision, labels, missing_mask)
    full, _ = compute_full_dlf_loss(
        model(text, audio, vision), labels, CRITERION, COSINE, HINGE
    )
    return full, (text, audio, vision, labels, None)


def freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def mode_names(mask):
    mapping = {(1, 1, 0): "LA", (1, 0, 1): "LV", (1, 0, 0): "L"}
    return [mapping[tuple(row)] for row in mask.detach().cpu().long().tolist()]


def build_compatibility(evaluator, loader, device, limit):
    records = []
    evaluator.eval()
    with torch.no_grad():
        for _, batch in iter_limited(loader, limit):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            values = {}
            for mode in MODES:
                mask = mode_to_mask(mode, text.size(0), device, audio.dtype)
                values[mode] = (
                    evaluator(text, audio, vision, mask)["output_logit"]
                    .view(-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            for offset, index in enumerate(batch["index"].view(-1).tolist()):
                records.append(
                    {
                        "train_index": int(index),
                        **{
                            "{}_prediction".format(mode): float(values[mode][offset])
                            for mode in MODES
                        },
                    }
                )
    frame = pd.DataFrame(records).sort_values("train_index", kind="mergesort")
    if frame.train_index.duplicated().any():
        raise RuntimeError("Compatibility cache duplicates train indices.")
    for mode in MISSING_MODES:
        delta = np.abs(
            frame.LAV_prediction.to_numpy()
            - frame["{}_prediction".format(mode)].to_numpy()
        )
        _, _, compatibility = compatibility_from_deltas(delta)
        frame["compat_{}".format(mode)] = compatibility
    return frame


def predict_expert(model, loader, device, expert_id, outer_fold_value, limit):
    model.eval()
    rows = []
    with torch.no_grad():
        for _, batch in iter_limited(loader, limit):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            label = batch["labels"]["M"].view(-1).numpy()
            identifiers = [str(value) for value in batch["id"]]
            indices = batch["index"].view(-1).numpy()
            for mode in MODES:
                mask = mode_to_mask(mode, text.size(0), device, audio.dtype)
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
                            "sample_id": sample_id,
                            "video_id": source_video(sample_id),
                            "train_index": int(indices[offset]),
                            "outer_fold": int(outer_fold_value),
                            "mode": mode,
                            "expert_id": expert_id,
                            "prediction": float(prediction[offset]),
                            "label": float(label[offset]),
                        }
                    )
    return pd.DataFrame(rows)


def clear_cuda(*models):
    for model in models:
        if model is not None:
            model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def main():
    cli = parse_args()
    protocol, protocol_sha = load_preregistered()
    splits, split_sha = load_split_manifest()
    if split_sha != protocol["source_split_sha256"]:
        raise RuntimeError("Source split SHA changed after preregistration.")
    output_group = "smoke" if cli.smoke_batches else "expert_oof"
    output = RESULT_ROOT / output_group / "outer_fold{}".format(cli.outer_fold)
    prediction_path = output / "oof_predictions.csv"
    final_manifest_path = output / "fold_manifest.json"
    if final_manifest_path.exists() and prediction_path.exists():
        manifest = json.loads(final_manifest_path.read_text())
        if manifest.get("prediction_sha256") == sha256_file(prediction_path):
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return

    setup_seed(23000 + int(cli.outer_fold))
    args = build_args(cli)
    args.seed = args.cur_seed = 23000 + int(cli.outer_fold)
    dataset = MMDataset(args, mode="train")
    args.seq_lens = dataset.get_seq_len()
    dataset_ids = [str(value) for value in dataset.ids]
    if ordered_id_sha(dataset_ids) != protocol["train_ordered_sample_id_sha256"]:
        raise RuntimeError("Runtime train sample order differs from preregistration.")

    outer_train = splits.outer_fold.astype(int) != int(cli.outer_fold)
    inner_column = "inner_valid_outer{}".format(cli.outer_fold)
    fit_indices = splits.loc[
        outer_train & (splits[inner_column].astype(int) == 0), "train_index"
    ].tolist()
    valid_indices = splits.loc[
        outer_train & (splits[inner_column].astype(int) == 1), "train_index"
    ].tolist()
    holdout_indices = splits.loc[~outer_train, "train_index"].tolist()
    if not fit_indices or not valid_indices or not holdout_indices:
        raise RuntimeError("Empty outer/inner source split.")
    fit_sources = set(splits.loc[splits.train_index.isin(fit_indices), "video_id"])
    valid_sources = set(splits.loc[splits.train_index.isin(valid_indices), "video_id"])
    holdout_sources = set(splits.loc[splits.train_index.isin(holdout_indices), "video_id"])
    if fit_sources & valid_sources or (fit_sources | valid_sources) & holdout_sources:
        raise RuntimeError("Source leakage across fit/inner-valid/outer-holdout.")

    def loader(indices, shuffle, seed):
        return build_subset_loader(
            dataset,
            indices,
            args.batch_size,
            shuffle,
            seed,
            cli.num_workers,
        )

    valid_loader = loader(valid_indices, False, 0)
    holdout_loader = loader(holdout_indices, False, 0)
    predictions = []
    clean_checkpoints = {}
    moddrop_checkpoints = {}

    # Train the seed-specific clean teachers only on the outer-training fold.
    for seed in (1111, 1114):
        setup_seed(seed)
        args.seed = args.cur_seed = seed
        train_loader = loader(fit_indices, True, seed)
        clean = DLF(args).to(args.device)
        missing_generator = torch.Generator().manual_seed(seed + 104729)

        def clean_step(model, batch, epoch, step):
            del epoch, step
            loss, _ = full_supervised_loss(
                model, batch, args.device, missing_generator, False
            )
            return loss, {}

        clean, clean_manifest = train_with_inner_selection(
            "clean_seed{}".format(seed),
            clean,
            train_loader,
            valid_loader,
            args,
            cli,
            clean_step,
            evaluate_clean,
            output / "components" / "clean_seed{}".format(seed),
        )
        clean_checkpoints[seed] = Path(clean_manifest["checkpoint"])
        clear_cuda(clean)

    # ModDrop experts, also retained as the frozen CF compatibility evaluators.
    for seed in (1111, 1114):
        setup_seed(seed)
        args.seed = args.cur_seed = seed
        train_loader = loader(fit_indices, True, seed)
        backbone = load_state(DLF(args), clean_checkpoints[seed], "cpu")
        model = MissingModalityWrapper(
            backbone, args.feature_dims[1], args.feature_dims[2]
        ).to(args.device)
        missing_generator = torch.Generator().manual_seed(seed + 104729)

        def moddrop_step(current, batch, epoch, step):
            del epoch, step
            loss, _ = full_supervised_loss(
                current, batch, args.device, missing_generator, True
            )
            return loss, {}

        model, manifest = train_with_inner_selection(
            "moddrop_seed{}".format(seed),
            model,
            train_loader,
            valid_loader,
            args,
            cli,
            moddrop_step,
            evaluate_missing,
            output / "experts" / "moddrop_seed{}".format(seed),
        )
        moddrop_checkpoints[seed] = Path(manifest["checkpoint"])
        predictions.append(
            predict_expert(
                model,
                holdout_loader,
                args.device,
                "moddrop_seed{}".format(seed),
                cli.outer_fold,
                cli.smoke_batches,
            )
        )
        clear_cuda(model)

    # Uniform prediction KD seed1111, matching the frozen Stage19 uniform loss.
    seed = 1111
    setup_seed(seed)
    args.seed = args.cur_seed = seed
    train_loader = loader(fit_indices, True, seed)
    teacher = freeze(load_state(DLF(args), clean_checkpoints[seed], "cpu").to(args.device))
    backbone = load_state(DLF(args), clean_checkpoints[seed], "cpu")
    uniform = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    missing_generator = torch.Generator().manual_seed(seed + 104729)

    def uniform_step(current, batch, epoch, step):
        del epoch, step
        supervised, tensors = full_supervised_loss(
            current, batch, args.device, missing_generator, True
        )
        text, audio, vision, _, missing_mask = tensors
        with torch.no_grad():
            target = teacher(text, audio, vision)["output_logit"]
        prediction = current(text, audio, vision, missing_mask)["output_logit"]
        kd = torch.nn.functional.smooth_l1_loss(prediction, target)
        return supervised + kd, {"kd": kd}

    uniform, manifest = train_with_inner_selection(
        "uniform_kd_seed1111",
        uniform,
        train_loader,
        valid_loader,
        args,
        cli,
        uniform_step,
        evaluate_missing,
        output / "experts" / "uniform_kd_seed1111",
    )
    predictions.append(
        predict_expert(
            uniform,
            holdout_loader,
            args.device,
            "uniform_kd_seed1111",
            cli.outer_fold,
            cli.smoke_batches,
        )
    )
    clear_cuda(uniform, teacher)

    # CFCompatKD experts use an evaluator and compatibility ranks trained only
    # on the same outer-training fit subset.
    for seed in (1111, 1114):
        setup_seed(seed)
        args.seed = args.cur_seed = seed
        cf_fit_indices = (
            fit_indices[: cli.smoke_batches * int(args.batch_size)]
            if cli.smoke_batches
            else fit_indices
        )
        cache_loader = loader(cf_fit_indices, False, 0)
        evaluator_backbone = DLF(args)
        evaluator = MissingModalityWrapper(
            evaluator_backbone, args.feature_dims[1], args.feature_dims[2]
        )
        load_state(evaluator, moddrop_checkpoints[seed], "cpu")
        evaluator = freeze(evaluator.to(args.device))
        compatibility = build_compatibility(
            evaluator, cache_loader, args.device, cli.smoke_batches
        )
        cache_path = output / "experts" / "cfcompat_seed{}".format(seed) / "train_only_compatibility.csv"
        atomic_csv(compatibility, cache_path)
        by_index = compatibility.set_index("train_index").to_dict("index")

        teacher = freeze(
            load_state(DLF(args), clean_checkpoints[seed], "cpu").to(args.device)
        )
        backbone = load_state(DLF(args), clean_checkpoints[seed], "cpu")
        model = MissingModalityWrapper(
            backbone, args.feature_dims[1], args.feature_dims[2]
        ).to(args.device)
        missing_generator = torch.Generator().manual_seed(seed + 104729)
        train_loader = loader(
            cf_fit_indices, not bool(cli.smoke_batches), seed
        )

        def cfcompat_step(current, batch, epoch, step):
            del epoch, step
            supervised, tensors = full_supervised_loss(
                current, batch, args.device, missing_generator, True
            )
            text, audio, vision, labels, missing_mask = tensors
            names = mode_names(missing_mask)
            indices = batch["index"].view(-1).tolist()
            gate = torch.as_tensor(
                [
                    float(by_index[int(index)]["compat_{}".format(mode)])
                    for index, mode in zip(indices, names)
                ],
                dtype=labels.dtype,
                device=args.device,
            )
            with torch.no_grad():
                target = teacher(text, audio, vision)["output_logit"]
            prediction = current(text, audio, vision, missing_mask)["output_logit"]
            kd, _ = gated_kd_loss(prediction, target, gate)
            return supervised + kd, {"kd": kd}

        model, manifest = train_with_inner_selection(
            "cfcompat_seed{}".format(seed),
            model,
            train_loader,
            valid_loader,
            args,
            cli,
            cfcompat_step,
            evaluate_missing,
            output / "experts" / "cfcompat_seed{}".format(seed),
        )
        predictions.append(
            predict_expert(
                model,
                holdout_loader,
                args.device,
                "cfcompat_seed{}".format(seed),
                cli.outer_fold,
                cli.smoke_batches,
            )
        )
        clear_cuda(model, teacher, evaluator)

    frame = pd.concat(predictions, ignore_index=True)
    expected_experts = set(EXPERTS)
    if set(frame.expert_id) != expected_experts:
        raise RuntimeError("OOF candidate pool differs from preregistration.")
    key = ["sample_id", "mode", "expert_id"]
    if frame.duplicated(key).any():
        raise RuntimeError("Duplicate OOF sample/mode/expert rows.")
    expected_samples = (
        min(len(holdout_indices), cli.smoke_batches * int(args.batch_size))
        if cli.smoke_batches
        else len(holdout_indices)
    )
    if len(frame) != expected_samples * len(MODES) * len(EXPERTS):
        raise RuntimeError("OOF prediction ledger is incomplete.")
    atomic_csv(frame.sort_values(key, kind="mergesort"), prediction_path)
    manifest = {
        "stage": "Stage23A Expert outer cross-fit",
        "outer_fold": int(cli.outer_fold),
        "expert_ids": list(EXPERTS),
        "fit_samples": len(fit_indices),
        "inner_valid_samples": len(valid_indices),
        "outer_holdout_samples": expected_samples,
        "fit_sources": len(fit_sources),
        "inner_valid_sources": len(valid_sources),
        "outer_holdout_sources": len(holdout_sources),
        "source_disjoint": True,
        "protocol_sha256": protocol_sha,
        "source_split_sha256": split_sha,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "ordered_holdout_sample_id_sha256": ordered_id_sha(
            frame.sample_id.drop_duplicates().tolist()
        ),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "test_loader_constructed": False,
        "student_trained": False,
        "code_commit": git_head(),
        "completed_at": utc_now(),
    }
    atomic_json(final_manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
