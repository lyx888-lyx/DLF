"""Memory-safe formal entry point for Safe-CFCompatKD.

The frozen ModDrop evaluator is used once before training to cache train
LA/LV/L predictions and Valid reference predictions.  It is then deleted so
formal optimization keeps only the original Teacher and Student on device.
"""
from __future__ import annotations

import gc

import numpy as np
import torch

import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_safe_projection_valid_screen_v2 as hardened
from train_cf_compat_kd import batch_to_device
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_for_modes,
    evaluator_prediction,
    gated_kd_loss,
)
from trains.singleTask.cfcompat_safe_projection_utils import safe_project_teacher
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.fixed_kd_utils import teacher_lav_prediction
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
)


_CURRENT_RUN = None
_ORIGINAL_LOAD_ASSETS = base.load_assets
_ORIGINAL_TRAIN_TRAJECTORY = base.train_trajectory


class FrozenPredictionCache:
    """CPU prediction cache with an empty parameter interface."""

    def __init__(self, train_by_index, valid_reference):
        self.train_by_index = train_by_index
        self.valid_reference = valid_reference

    def parameters(self):
        return iter(())


def build_train_baseline_cache(evaluator, args, num_workers):
    loader = build_single_split_loader(args, "train", num_workers)
    by_index = {}
    with preserve_rng_state():
        for batch in loader:
            text, audio, vision, _ = batch_to_device(batch, args.device)
            predictions = {
                mode: evaluator_prediction(
                    evaluator, text, audio, vision, mode
                ).view(-1)
                for mode in MISSING_MODES
            }
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            for offset, index in enumerate(indices):
                if int(index) in by_index:
                    raise RuntimeError(
                        "Frozen baseline cache contains a duplicate train index."
                    )
                by_index[int(index)] = {
                    mode: float(predictions[mode][offset].detach().cpu())
                    for mode in MISSING_MODES
                }
    if len(by_index) != 1284 or set(by_index) != set(range(1284)):
        raise RuntimeError(
            "Frozen baseline cache must contain train indices 0..1283 exactly."
        )
    return by_index


def load_assets(cli, args, loaders, seed):
    teacher, student, evaluator, assets = _ORIGINAL_LOAD_ASSETS(
        cli, args, loaders, seed
    )
    with preserve_rng_state():
        valid_reference = hardened.reference_prediction_rows(
            evaluator, teacher, loaders["valid"], args.device
        )
    if len(valid_reference) != 229:
        raise RuntimeError("MOSI Valid reference cache must contain 229 samples.")

    train_by_index = {}
    if _CURRENT_RUN in ("safe_uniform", "safe_cfcompat"):
        train_by_index = build_train_baseline_cache(
            evaluator, args, cli.num_workers
        )
    elif _CURRENT_RUN != "cfcompat_replay":
        raise RuntimeError("Safe-CFCompat current run was not bound.")

    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assets["baseline_train_cache_sample_count"] = int(len(train_by_index))
    assets["baseline_valid_cache_sample_count"] = int(len(valid_reference))
    return (
        teacher,
        student,
        FrozenPredictionCache(train_by_index, valid_reference),
        assets,
    )


def cached_baseline_prediction(bundle, indices, modes, device, dtype):
    values = []
    for index, mode in zip(indices, modes):
        if int(index) not in bundle.train_by_index:
            raise KeyError("Train baseline cache lacks sample {}.".format(index))
        if str(mode) not in bundle.train_by_index[int(index)]:
            raise KeyError(
                "Train baseline cache lacks mode {} for sample {}.".format(
                    mode, index
                )
            )
        values.append(bundle.train_by_index[int(index)][str(mode)])
    tensor = torch.as_tensor(values, device=device, dtype=dtype).view(-1, 1)
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Cached frozen baseline prediction is non-finite.")
    return tensor


def forward_objective(
    run,
    batch,
    missing_mask,
    modes,
    args,
    teacher,
    evaluator_bundle,
    student,
    cache_by_index,
    criterion,
    cosine,
    hinge,
):
    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = missing_mask.to(device=args.device, dtype=audio.dtype)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(
        full_output, labels, criterion, cosine, hinge
    )
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    teacher_prediction = teacher_lav_prediction(
        teacher, text, audio, vision
    )
    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    compatibility = compatibility_for_modes(
        cache_by_index,
        indices,
        list(modes),
        args.device,
        labels.dtype,
    ).view(-1)

    projection_records = []
    if run == "cfcompat_replay":
        kd_target = teacher_prediction.detach().view(-1, 1)
        gate = compatibility
        baseline_prediction = None
    else:
        baseline_prediction = cached_baseline_prediction(
            evaluator_bundle,
            indices,
            modes,
            args.device,
            labels.dtype,
        )
        kd_target, projection = safe_project_teacher(
            baseline_prediction, teacher_prediction, labels
        )
        gate = (
            torch.ones_like(compatibility)
            if run == "safe_uniform"
            else compatibility
        )
        for offset in range(labels.size(0)):
            projection_records.append(
                {
                    key: (
                        bool(value[offset].detach().cpu())
                        if value.dtype == torch.bool
                        else float(value[offset].detach().cpu())
                    )
                    for key, value in projection.items()
                }
            )

    kd_loss, each_kd = gated_kd_loss(
        missing_output["output_logit"], kd_target, gate
    )
    total_loss = full_loss + missing_loss + kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in cached Safe-CFCompat objective.")

    denominator = float(gate.sum().detach().cpu()) + 1e-8
    diagnostics = {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "mean_gate": float(gate.detach().mean().cpu()),
        "weighted_kd": float(
            (gate.detach() * each_kd.detach().view(-1)).sum().cpu()
            / denominator
        ),
        "baseline_missing_MAE": (
            float(
                torch.abs(
                    baseline_prediction.view(-1) - labels.view(-1)
                ).mean().cpu()
            )
            if baseline_prediction is not None
            else float("nan")
        ),
        "safe_target_MAE": (
            float(
                torch.abs(kd_target.view(-1) - labels.view(-1)).mean().cpu()
            )
            if baseline_prediction is not None
            else float("nan")
        ),
    }
    return total_loss, diagnostics, projection_records


def reference_prediction_rows(evaluator_bundle, teacher, loader, device):
    del teacher, loader, device
    return evaluator_bundle.valid_reference.copy()


def train_trajectory(cli, logger, output_root, model_root, seed, run):
    global _CURRENT_RUN
    if _CURRENT_RUN is not None:
        raise RuntimeError("Nested Safe-CFCompat trajectory binding is forbidden.")
    _CURRENT_RUN = str(run)
    try:
        return _ORIGINAL_TRAIN_TRAJECTORY(
            cli, logger, output_root, model_root, seed, run
        )
    finally:
        _CURRENT_RUN = None


def main():
    base.load_assets = load_assets
    base.forward_objective = forward_objective
    base.reference_prediction_rows = reference_prediction_rows
    base.train_trajectory = train_trajectory
    base.main()


if __name__ == "__main__":
    main()
