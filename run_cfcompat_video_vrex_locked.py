"""Locked Video-VREx entrypoint with one-pass test prediction export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train_cfcompat_video_vrex as base
from data_loader import MMDataLoader
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    build_single_split_loader,
    mode_to_mask,
    regression_metrics,
    validation_objective,
)


def evaluate_and_predictions(model, loader, device, criterion):
    """Evaluate all four modality states in exactly one loader traversal."""
    model.eval()
    modes = ("LAV",) + MISSING_MODES
    collected = {
        mode: {"prediction": [], "label": [], "loss": []}
        for mode in modes
    }
    rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = base.batch_to_device(batch, device)
            predictions = {}
            for mode in modes:
                mask = mode_to_mask(
                    mode, labels.size(0), device, audio.dtype
                )
                output = model(text, audio, vision, mask)["output_logit"]
                predictions[mode] = output
                collected[mode]["prediction"].append(output.detach().cpu())
                collected[mode]["label"].append(labels.detach().cpu())
                collected[mode]["loss"].append(
                    float(criterion(output, labels).detach().cpu())
                )
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            sample_ids = list(batch["id"])
            for offset, index in enumerate(indices):
                rows.append(
                    {
                        "sample_id": str(sample_ids[offset]),
                        "sample_index": int(index),
                        "label": float(labels[offset].item()),
                        **{
                            "{}_pred".format(mode): float(
                                predictions[mode][offset].item()
                            )
                            for mode in modes
                        },
                    }
                )
    metrics = {}
    for mode, values in collected.items():
        prediction = torch.cat(values["prediction"], dim=0)
        labels = torch.cat(values["label"], dim=0)
        row = regression_metrics(prediction, labels)
        row["Loss"] = float(np.mean(values["loss"]))
        metrics[mode] = row
    frame = pd.DataFrame(rows).sort_values(
        "sample_index", kind="mergesort"
    )
    return metrics, frame


def one_pass_selected_test(cli, args, selected, baseline, logger):
    test_loader = build_single_split_loader(args, "test", cli.num_workers)
    valid_loaders = MMDataLoader(args, cli.num_workers)
    teacher, student, _ = base.load_assets(cli, args, valid_loaders)
    del teacher
    student.load_state_dict(
        torch.load(selected["MainCheckpoint"], map_location=args.device),
        strict=True,
    )
    criterion = nn.L1Loss()
    test_metrics, test_predictions = evaluate_and_predictions(
        student, test_loader, args.device, criterion
    )
    candidate = dict(selected)
    candidate["J_test_at_valid_best"] = float(
        validation_objective(test_metrics)
    )
    candidate.update(base._flatten(test_metrics, "test_at_valid_best"))
    candidate["TestConstructed"] = True
    candidate["TestLoaderConstructionCount"] = 1
    candidate["TestLoaderTraversalCount"] = 1
    gate = base.candidate_test_gate(candidate, baseline)
    logger.info(
        "selected lambda=%.6g frozen_test_J=%.6f gain=%+.6f passed=%s",
        candidate["LambdaVREx"],
        candidate["J_test_at_valid_best"],
        gate["gain_test_J"],
        gate["passed"],
    )
    return candidate, gate, test_predictions


def finalize_immutable_domain_audit(cli):
    result_root, _ = base.result_paths(cli)
    immutable = result_root / "domain_audit" / "video_domain_audit.json"
    manifest_path = result_root / "video_vrex_source_manifest.json"
    if not immutable.is_file():
        raise FileNotFoundError(
            "Immutable per-seed domain audit is absent: {}".format(immutable)
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Video-VREx source manifest is absent: {}".format(manifest_path)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["domain_audit"] = str(immutable)
    manifest["domain_audit_sha256"] = checkpoint_sha256(immutable)
    manifest["test_loader_traversal_count"] = (
        1 if manifest["test_constructed_after_valid_gate"] else 0
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    base.evaluate_selected_test = one_pass_selected_test
    base.main()
    cli = base.parse_args()
    finalize_immutable_domain_audit(cli)


if __name__ == "__main__":
    main()
