"""Hardened entry point for Safe-CFCompatKD evaluator inference."""
from __future__ import annotations

import pandas as pd
import torch

import train_cfcompat_safe_projection_valid_screen as base
from train_cf_compat_kd import batch_to_device
from trains.singleTask.cf_compat_kd_utils import evaluator_prediction
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.fixed_kd_utils import (
    _restore_normal_position_cache,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import MISSING_MODES


def frozen_baseline_prediction(
    evaluator,
    text,
    audio,
    vision,
    missing_mask,
):
    """Run a mixed-mask frozen evaluator without RNG or inference-cache drift."""
    with preserve_rng_state():
        with torch.inference_mode():
            prediction = evaluator(
                text, audio, vision, missing_mask
            )["output_logit"]
        normal = prediction.detach().clone()
        _restore_normal_position_cache()
    return normal.view(-1, 1)


def reference_prediction_rows(evaluator, teacher, loader, device):
    """Create deterministic Valid references using cache-safe helpers."""
    evaluator.eval()
    teacher.eval()
    rows = []
    for batch in loader:
        text, audio, vision, labels = batch_to_device(batch, device)
        teacher_prediction = teacher_lav_prediction(
            teacher, text, audio, vision
        ).view(-1)
        baseline = {
            mode: evaluator_prediction(
                evaluator, text, audio, vision, mode
            ).view(-1)
            for mode in ("LAV",) + MISSING_MODES
        }
        indices = batch["index"].view(-1).cpu().numpy().astype(int)
        identifiers = list(batch["id"])
        for offset, index in enumerate(indices):
            row = {
                "sample_index": int(index),
                "sample_id": str(identifiers[offset]),
                "label": float(labels[offset].item()),
                "teacher_prediction": float(
                    teacher_prediction[offset].detach().cpu()
                ),
            }
            for mode in ("LAV",) + MISSING_MODES:
                row[f"baseline_{mode}_pred"] = float(
                    baseline[mode][offset].detach().cpu()
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "sample_index", kind="mergesort"
    )


def main():
    base._frozen_baseline_prediction = frozen_baseline_prediction
    base.reference_prediction_rows = reference_prediction_rows
    base.main()


if __name__ == "__main__":
    main()
