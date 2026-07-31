"""Nested grouped OOF CFCompatKD cache orchestration for V9.2."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _checkpoint_payload,
    _load_state,
    _mae,
    predict_wrapper_lav,
    train_cfcompat_student,
    train_clean_dlf,
    train_moddrop_evaluator,
)
from .oof_group_splits_v92 import (
    StageLimits,
    build_nested_group_folds,
    build_subset_loader,
    canonical_sample_id,
    conversation_group_id,
)

logger = logging.getLogger("MMSA")
CACHE_VERSION = "nested_grouped_oof_cfcompat_v9_2"


def run_nested_oof_cfcompat(
    args,
    train_dataset,
    output_dir: Path,
    seed: int,
    outer_folds: int,
    inner_valid_fraction: float,
    num_workers: int,
    limits: StageLimits,
    resume: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = list(train_dataset.ids)
    labels = np.asarray(
        train_dataset.labels["M"], dtype=np.float32
    ).reshape(-1)
    specs, manifest = build_nested_group_folds(
        sample_ids,
        labels,
        outer_folds=outer_folds,
        inner_valid_fraction=inner_valid_fraction,
        seed=seed,
    )
    manifest.to_csv(
        output_dir / "v92_nested_group_manifest.csv", index=False
    )

    sample_count = len(train_dataset)
    oof_prediction = torch.full(
        (sample_count, 1), float("nan"), dtype=torch.float32
    )
    oof_feature = None
    fold_index = torch.full((sample_count,), -1, dtype=torch.long)
    fold_metadata = []

    for spec in specs:
        fold_dir = output_dir / f"outer_fold_{spec.outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_seed = int(seed) + 1009 * (spec.outer_fold + 1)
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)

        train_loader = build_subset_loader(
            train_dataset,
            spec.inner_train_indices,
            int(args.batch_size),
            num_workers,
            True,
            fold_seed,
        )
        valid_loader = build_subset_loader(
            train_dataset,
            spec.inner_valid_indices,
            int(args.batch_size),
            num_workers,
            False,
            fold_seed,
        )
        holdout_loader = build_subset_loader(
            train_dataset,
            spec.outer_holdout_indices,
            int(args.batch_size),
            num_workers,
            False,
            fold_seed,
        )

        clean_checkpoint = fold_dir / "clean_dlf_best_inner_valid.pth"
        moddrop_checkpoint = (
            fold_dir / "moddrop_evaluator_best_inner_valid.pth"
        )
        cfcompat_checkpoint = (
            fold_dir / "cfcompat_student_best_inner_valid.pth"
        )
        fold_cache = fold_dir / "outer_holdout_cache.pth"

        if resume and fold_cache.is_file():
            cached = torch.load(fold_cache, map_location="cpu")
            holdout_rows = cached["rows"]
            stage_summary = cached["stage_summary"]
            logger.info(
                "V9.2 fold=%d resumed cached outer holdout",
                spec.outer_fold,
            )
        else:
            clean_summary = train_clean_dlf(
                args,
                train_loader,
                valid_loader,
                clean_checkpoint,
                limits,
            )
            pd.DataFrame(clean_summary["history"]).to_csv(
                fold_dir / "clean_history.csv", index=False
            )
            moddrop_summary = train_moddrop_evaluator(
                args,
                train_loader,
                valid_loader,
                clean_checkpoint,
                moddrop_checkpoint,
                limits,
                fold_seed,
            )
            pd.DataFrame(moddrop_summary["history"]).to_csv(
                fold_dir / "moddrop_history.csv", index=False
            )
            cfcompat_summary = train_cfcompat_student(
                args,
                train_loader,
                valid_loader,
                clean_checkpoint,
                moddrop_checkpoint,
                cfcompat_checkpoint,
                limits,
                fold_seed,
            )
            pd.DataFrame(cfcompat_summary["history"]).to_csv(
                fold_dir / "cfcompat_history.csv", index=False
            )

            backbone = DLF(args).to(args.device)
            student = MissingModalityWrapper(
                backbone,
                int(args.feature_dims[1]),
                int(args.feature_dims[2]),
            ).to(args.device)
            student.load_state_dict(
                _load_state(cfcompat_checkpoint, args.device), strict=True
            )
            holdout_rows = predict_wrapper_lav(
                student,
                holdout_loader,
                args.device,
                capture_features=True,
            )
            stage_summary = {
                "clean": {
                    key: value
                    for key, value in clean_summary.items()
                    if key != "history"
                },
                "moddrop": {
                    key: value
                    for key, value in moddrop_summary.items()
                    if key != "history"
                },
                "cfcompat": {
                    key: value
                    for key, value in cfcompat_summary.items()
                    if key != "history"
                },
                "checkpoints": {
                    "clean": _checkpoint_payload(clean_checkpoint),
                    "moddrop": _checkpoint_payload(moddrop_checkpoint),
                    "cfcompat": _checkpoint_payload(cfcompat_checkpoint),
                },
            }
            torch.save(
                {"rows": holdout_rows, "stage_summary": stage_summary},
                fold_cache,
            )
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        expected = set(spec.outer_holdout_indices)
        observed = {
            int(row["sample_index"]) for row in holdout_rows
        }
        if expected != observed:
            raise RuntimeError(
                f"outer fold {spec.outer_fold} cache differs from manifest"
            )
        for row in holdout_rows:
            index = int(row["sample_index"])
            feature = row["feature"].view(-1).float()
            if oof_feature is None:
                oof_feature = torch.full(
                    (sample_count, feature.numel()),
                    float("nan"),
                    dtype=torch.float32,
                )
            if feature.numel() != oof_feature.size(1):
                raise RuntimeError(
                    "OOF feature dimension changed across folds"
                )
            oof_prediction[index, 0] = float(row["prediction"])
            oof_feature[index] = feature
            fold_index[index] = int(spec.outer_fold)

        fold_metadata.append(
            {
                "outer_fold": int(spec.outer_fold),
                "fold_seed": fold_seed,
                "inner_train_count": len(spec.inner_train_indices),
                "inner_valid_count": len(spec.inner_valid_indices),
                "outer_holdout_count": len(spec.outer_holdout_indices),
                "inner_train_group_count": len(spec.inner_train_groups),
                "inner_valid_group_count": len(spec.inner_valid_groups),
                "outer_holdout_group_count": len(
                    spec.outer_holdout_groups
                ),
                "stage_summary": stage_summary,
            }
        )

    if oof_feature is None:
        raise RuntimeError("OOF feature cache was not constructed")
    if (
        not torch.isfinite(oof_prediction).all()
        or not torch.isfinite(oof_feature).all()
    ):
        raise RuntimeError("OOF cache contains non-finite values")
    if bool((fold_index < 0).any()):
        raise RuntimeError("OOF cache has unassigned fold indices")

    group_ids = [conversation_group_id(value) for value in sample_ids]
    labels_tensor = torch.tensor(
        labels, dtype=torch.float32
    ).view(-1, 1)
    payload = {
        "version": CACHE_VERSION,
        "method": "nested_grouped_oof_cfcompat",
        "dataset": str(args.dataset_name),
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "inner_valid_fraction": float(inner_valid_fraction),
        "sample_ids": [
            canonical_sample_id(value) for value in sample_ids
        ],
        "group_ids": group_ids,
        "labels": labels_tensor,
        "oof_prediction": oof_prediction,
        "oof_feature": oof_feature,
        "fold_index": fold_index,
        "fold_metadata": fold_metadata,
        "protocol": (
            "Each outer holdout group is excluded from clean DLF, "
            "ModDrop evaluator, compatibility cache, CFCompatKD "
            "optimization, and checkpoint selection. Selection uses a "
            "disjoint inner validation group split."
        ),
    }
    cache_path = (
        output_dir / "nested_grouped_oof_cfcompat_cache_v92.pth"
    )
    torch.save(payload, cache_path)
    pd.DataFrame(
        {
            "sample_index": np.arange(sample_count),
            "sample_id": payload["sample_ids"],
            "group_id": group_ids,
            "outer_fold": fold_index.tolist(),
            "label": labels_tensor.view(-1).tolist(),
            "oof_prediction": oof_prediction.view(-1).tolist(),
            "oof_residual": (
                labels_tensor - oof_prediction
            ).view(-1).tolist(),
        }
    ).to_csv(
        output_dir / "nested_grouped_oof_cfcompat_predictions_v92.csv",
        index=False,
    )
    summary = {
        "version": CACHE_VERSION,
        "dataset": str(args.dataset_name),
        "seed": int(seed),
        "outer_folds": int(outer_folds),
        "inner_valid_fraction": float(inner_valid_fraction),
        "sample_count": sample_count,
        "group_count": len(set(group_ids)),
        "feature_dim": int(oof_feature.size(1)),
        "oof_mae": _mae(oof_prediction, labels_tensor),
        "fold_metadata": fold_metadata,
        "cache_path": str(cache_path),
        "protocol": payload["protocol"],
    }
    (
        output_dir / "nested_grouped_oof_cfcompat_summary_v92.json"
    ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return payload
