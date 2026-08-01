"""Strict outer-fold-isolated V9.3-architecture shadow expert pool for V9.8.

The final deployment experts remain the original frozen V9.3 pool. This module
only constructs the Train OOF shadow pool used to supervise the frontier coach.
Every upstream checkpoint used for an outer holdout prediction comes from the
matching V9.2 outer-fold directory and was trained/selected without that complete
holdout group.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import torch

from .cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _load_state,
    predict_plain,
    predict_wrapper_lav,
)
from .frontier_expert_pool_v97 import (
    RoleCrossfitConfigV97,
    TailCrossfitConfigV97,
    _collect_role_expert,
    _collect_tail_expert,
    _fit_role_expert,
    _fit_tail_expert,
)
from .function_space_features_v92 import (
    FEATURE_SPACE_VERSION,
    predict_wrapper_function_space,
)
from .model.AttainableFrontierCoachV97 import (
    ACTION_NAMES,
    SPECIALIST_NAMES,
    attainable_frontier_targets,
    stack_action_predictions,
)
from .oof_group_splits_v92 import (
    build_nested_group_folds,
    build_subset_loader,
    canonical_sample_id,
    conversation_group_id,
)
from .role_conditioned_experts_v9 import fit_role_teacher_weights

logger = logging.getLogger("MMSA")
POOL_VERSION = "v98_strict_nested_v93_architecture_frontier_pool"
TEACHER_NAMES = ("clean", "moddrop", "cfcompat")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_ids(values: Sequence[object]) -> list[str]:
    return [canonical_sample_id(value) for value in values]


def _dataset_ids(dataset) -> list[str]:
    if not hasattr(dataset, "ids"):
        raise AttributeError("MMDataset must expose ids for V9.8 alignment")
    return _canonical_ids(list(dataset.ids))


def _index_map(values: Sequence[object], name: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for index, value in enumerate(values):
        key = canonical_sample_id(value)
        if key in result:
            raise RuntimeError(f"duplicate {name} sample id: {key}")
        result[key] = int(index)
    return result


def _fold_checkpoint_paths(top_oof_cache_path: Path, fold: int) -> Dict[str, Path]:
    fold_dir = Path(top_oof_cache_path).parent / f"outer_fold_{int(fold)}"
    paths = {
        "clean": fold_dir / "clean_dlf_best_inner_valid.pth",
        "moddrop": fold_dir / "moddrop_evaluator_best_inner_valid.pth",
        "cfcompat": fold_dir / "cfcompat_student_best_inner_valid.pth",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Strict V9.8 requires the saved V9.2 fold-local upstream "
            f"checkpoints; missing={missing}"
        )
    return paths


def _new_teacher_model(args, name: str, checkpoint: Path):
    if name == "clean":
        model = DLF(args).to(args.device)
        model.load_state_dict(_load_state(checkpoint, args.device), strict=True)
        return model, False
    if name in {"moddrop", "cfcompat"}:
        model = MissingModalityWrapper(
            DLF(args).to(args.device),
            int(args.feature_dims[1]),
            int(args.feature_dims[2]),
        ).to(args.device)
        model.load_state_dict(_load_state(checkpoint, args.device), strict=True)
        return model, True
    raise ValueError(f"unknown strict teacher name: {name}")


def _rows_by_index(rows: Sequence[Mapping[str, object]]) -> Dict[int, Mapping[str, object]]:
    result: Dict[int, Mapping[str, object]] = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in result:
            raise RuntimeError(f"duplicate prediction row for sample index {index}")
        result[index] = row
    return result


def _collect_teacher_predictions(
    args,
    train_dataset,
    indices: Sequence[int],
    checkpoint_paths: Mapping[str, Path],
    batch_size: int,
    num_workers: int,
):
    loader = build_subset_loader(
        train_dataset,
        indices,
        batch_size,
        num_workers,
        False,
        0,
    )
    ordered = [int(value) for value in indices]
    predictions = []
    sample_ids = None
    labels = None
    for name in TEACHER_NAMES:
        model, wrapper = _new_teacher_model(args, name, checkpoint_paths[name])
        rows = (
            predict_wrapper_lav(model, loader, args.device, False)
            if wrapper
            else predict_plain(model, loader, args.device, False)
        )
        mapping = _rows_by_index(rows)
        if set(mapping) != set(ordered):
            raise RuntimeError(
                f"{name} teacher prediction set differs from requested indices"
            )
        local_ids = [
            canonical_sample_id(mapping[index]["sample_id"]) for index in ordered
        ]
        local_labels = torch.tensor(
            [float(mapping[index]["label"]) for index in ordered],
            dtype=torch.float32,
        ).view(-1, 1)
        local_predictions = torch.tensor(
            [float(mapping[index]["prediction"]) for index in ordered],
            dtype=torch.float32,
        ).view(-1, 1)
        if sample_ids is None:
            sample_ids = local_ids
            labels = local_labels
        elif sample_ids != local_ids or not torch.allclose(labels, local_labels):
            raise RuntimeError(
                "strict fold-local teacher alignment changed across candidates"
            )
        predictions.append(local_predictions)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "predictions": torch.stack(predictions, dim=1),
    }


def _collect_fold_cfcompat_space(
    args,
    train_dataset,
    indices: Sequence[int],
    checkpoint: Path,
    batch_size: int,
    num_workers: int,
):
    loader = build_subset_loader(
        train_dataset,
        indices,
        batch_size,
        num_workers,
        False,
        0,
    )
    model, wrapper = _new_teacher_model(args, "cfcompat", checkpoint)
    if not wrapper:
        raise AssertionError("CFCompat checkpoint must use MissingModalityWrapper")
    rows = predict_wrapper_function_space(model, loader, args.device, True)
    mapping = _rows_by_index(rows)
    ordered = [int(value) for value in indices]
    if set(mapping) != set(ordered):
        raise RuntimeError(
            "fold-local CFCompat feature set differs from requested indices"
        )
    result = {
        "sample_ids": [
            canonical_sample_id(mapping[index]["sample_id"]) for index in ordered
        ],
        "labels": torch.tensor(
            [float(mapping[index]["label"]) for index in ordered],
            dtype=torch.float32,
        ).view(-1, 1),
        "anchor": torch.tensor(
            [float(mapping[index]["prediction"]) for index in ordered],
            dtype=torch.float32,
        ).view(-1, 1),
        "feature": torch.stack(
            [mapping[index]["feature"].float().view(-1) for index in ordered],
            dim=0,
        ),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _reconstruct_outer_specs(top):
    specs, manifest = build_nested_group_folds(
        top["sample_ids"],
        top["labels"].view(-1).tolist(),
        outer_folds=int(top["outer_folds"]),
        inner_valid_fraction=float(top["inner_valid_fraction"]),
        seed=int(top["seed"]),
    )
    reconstructed = torch.full_like(top["fold_index"], -1)
    for spec in specs:
        reconstructed[list(spec.outer_holdout_indices)] = int(spec.outer_fold)
    if not torch.equal(reconstructed.cpu(), top["fold_index"].cpu()):
        raise RuntimeError(
            "V9.8 reconstructed folds do not match the V9.2 OOF cache"
        )
    return specs, manifest


def _build_fold_teacher_artifacts(
    args,
    train_dataset,
    development_indices: Sequence[int],
    inner_valid_indices: Sequence[int],
    checkpoint_paths: Mapping[str, Path],
    batch_size: int,
    num_workers: int,
    teacher_fit_steps: int,
    min_region_samples: int,
):
    development = _collect_teacher_predictions(
        args,
        train_dataset,
        development_indices,
        checkpoint_paths,
        batch_size,
        num_workers,
    )
    valid = _collect_teacher_predictions(
        args,
        train_dataset,
        inner_valid_indices,
        checkpoint_paths,
        batch_size,
        num_workers,
    )
    valid_mae = torch.abs(
        valid["predictions"] - valid["labels"].unsqueeze(1)
    ).mean(dim=(0, 2))
    anchor_index = int(valid_mae.argmin().item())
    fit = fit_role_teacher_weights(
        valid["predictions"],
        valid["labels"],
        valid["sample_ids"],
        steps=int(teacher_fit_steps),
        min_region_samples=int(min_region_samples),
    )
    return {
        "cache": {
            "splits": {
                "train": {
                    "sample_ids": development["sample_ids"],
                    "labels": development["labels"],
                    "predictions": development["predictions"],
                }
            }
        },
        "teacher_paths": [checkpoint_paths[name] for name in TEACHER_NAMES],
        "anchor_index": anchor_index,
        "global_weights": fit["global_weights"].float(),
        "role_weights": fit["role_weights"].float(),
        "valid_teacher_mae": {
            name: float(valid_mae[index].item())
            for index, name in enumerate(TEACHER_NAMES)
        },
        "selected_regularizations": fit["selected_regularizations"],
        "region_counts": fit["region_counts"],
    }


def build_strict_nested_v93_frontier_pool(
    args,
    train_dataset,
    top_oof_cache_path: Path,
    output_dir: Path,
    role_config: RoleCrossfitConfigV97,
    tail_config: TailCrossfitConfigV97,
    num_workers: int = 1,
    teacher_fit_steps: int = 300,
    min_region_samples: int = 10,
    resume: bool = True,
):
    """Build a strict top-outer-fold-isolated V9.3-architecture shadow pool."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "strict_nested_v93_frontier_pool_v98.pth"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu")
        if (
            payload.get("version") == POOL_VERSION
            and payload.get("role_config") == role_config.__dict__
            and payload.get("tail_config") == tail_config.__dict__
        ):
            logger.info("Reusing strict V9.8 frontier pool: %s", final_path)
            return payload

    top_oof_cache_path = Path(top_oof_cache_path)
    top = torch.load(top_oof_cache_path, map_location="cpu")
    required = {
        "sample_ids",
        "group_ids",
        "labels",
        "oof_prediction",
        "oof_feature",
        "fold_index",
        "feature_space",
        "outer_folds",
        "inner_valid_fraction",
        "seed",
    }
    if not required.issubset(top):
        raise ValueError(f"V9.2 OOF cache missing: {sorted(required - set(top))}")
    if top.get("feature_space") != FEATURE_SPACE_VERSION:
        raise ValueError("V9.8 requires aligned V9.2 function-space features")

    specs, manifest = _reconstruct_outer_specs(top)
    manifest.to_csv(output_dir / "v98_strict_nested_manifest.csv", index=False)
    top_ids = _canonical_ids(top["sample_ids"])
    top_id_map = _index_map(top_ids, "top-OOF")
    dataset_ids = _dataset_ids(train_dataset)
    dataset_id_map = _index_map(dataset_ids, "train-dataset")
    missing = sorted(set(top_ids) - set(dataset_id_map))
    if missing:
        raise RuntimeError(f"Train dataset misses {len(missing)} top OOF ids")
    top_to_dataset = [dataset_id_map[value] for value in top_ids]

    n = len(top_ids)
    predictions = torch.full((n, 4, 1), float("nan"))
    confidences = torch.full_like(predictions, float("nan"))
    corrections = torch.full_like(predictions, float("nan"))
    histories = []
    fold_rows = []
    name_to_column = {
        name: index for index, name in enumerate(SPECIALIST_NAMES)
    }

    for spec in specs:
        fold = int(spec.outer_fold)
        fold_dir = output_dir / f"outer_fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_path = fold_dir / "strict_nested_holdout_v98.pth"
        checkpoint_paths = _fold_checkpoint_paths(top_oof_cache_path, fold)
        checkpoint_sha = {
            name: _sha256(path) for name, path in checkpoint_paths.items()
        }
        holdout_top = torch.tensor(
            spec.outer_holdout_indices, dtype=torch.long
        )
        development_top = torch.tensor(
            sorted((*spec.inner_train_indices, *spec.inner_valid_indices)),
            dtype=torch.long,
        )
        if resume and fold_path.is_file():
            cached = torch.load(fold_path, map_location="cpu")
            if (
                cached.get("version") == POOL_VERSION
                and cached.get("role_config") == role_config.__dict__
                and cached.get("tail_config") == tail_config.__dict__
                and cached.get("checkpoint_sha256") == checkpoint_sha
            ):
                predictions[holdout_top] = cached["predictions"]
                confidences[holdout_top] = cached["confidences"]
                corrections[holdout_top] = cached["corrections"]
                histories.extend(cached.get("history", []))
                fold_rows.append(cached["metadata"])
                logger.info(
                    "V9.8 strict frontier outer fold=%d resumed", fold
                )
                continue

        development_dataset_indices = [
            top_to_dataset[index] for index in development_top.tolist()
        ]
        inner_valid_dataset_indices = [
            top_to_dataset[index] for index in spec.inner_valid_indices
        ]
        holdout_dataset_indices = [
            top_to_dataset[index] for index in holdout_top.tolist()
        ]
        teacher = _build_fold_teacher_artifacts(
            args,
            train_dataset,
            development_dataset_indices,
            inner_valid_dataset_indices,
            checkpoint_paths,
            role_config.batch_size,
            num_workers,
            teacher_fit_steps,
            min_region_samples,
        )
        development_space = _collect_fold_cfcompat_space(
            args,
            train_dataset,
            development_dataset_indices,
            checkpoint_paths["cfcompat"],
            tail_config.batch_size,
            num_workers,
        )
        expected_dev_ids = [
            top_ids[index] for index in development_top.tolist()
        ]
        if development_space["sample_ids"] != expected_dev_ids:
            raise RuntimeError(
                "fold-local CFCompat development order is misaligned"
            )

        fold_predictions = torch.full(
            (len(holdout_top), 4, 1), float("nan")
        )
        fold_confidences = torch.full_like(
            fold_predictions, float("nan")
        )
        fold_corrections = torch.full_like(
            fold_predictions, float("nan")
        )
        holdout_position = {
            int(top_index): offset
            for offset, top_index in enumerate(holdout_top.tolist())
        }
        fold_history = []

        for offset, role in enumerate(("boundary", "positive")):
            role_seed = (
                int(args.seed)
                + 31013 * (fold + 1)
                + 101 * (offset + 1)
            )
            model, history = _fit_role_expert(
                args,
                train_dataset,
                development_dataset_indices,
                role,
                teacher,
                role_config,
                role_seed,
                num_workers,
            )
            rows = _collect_role_expert(
                model,
                args,
                train_dataset,
                holdout_dataset_indices,
                role,
                top_id_map,
                role_config.batch_size,
                num_workers,
            )
            column = name_to_column[role]
            for row in rows:
                local = holdout_position[int(row["top_index"])]
                fold_predictions[local, column] = row["prediction"].view(1)
                fold_confidences[local, column] = row["confidence"].view(1)
                fold_corrections[local, column] = row["correction"].view(1)
            tagged = [{"outer_fold": fold, **row} for row in history]
            histories.extend(tagged)
            fold_history.extend(tagged)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for offset, role in enumerate(
            ("strong_negative", "strong_positive")
        ):
            tail_seed = (
                int(args.seed)
                + 41017 * (fold + 1)
                + 131 * (offset + 1)
            )
            model, history = _fit_tail_expert(
                development_space["feature"],
                development_space["anchor"],
                development_space["labels"],
                role,
                tail_config,
                args.device,
                tail_seed,
            )
            collected = _collect_tail_expert(
                model,
                top["oof_feature"][holdout_top],
                top["oof_prediction"][holdout_top],
                args.device,
                tail_config.batch_size,
            )
            column = name_to_column[role]
            fold_predictions[:, column] = collected["prediction"]
            fold_confidences[:, column] = collected[
                "applicability_prob"
            ]
            fold_corrections[:, column] = collected["correction"]
            tagged = [{"outer_fold": fold, **row} for row in history]
            histories.extend(tagged)
            fold_history.extend(tagged)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if any(
            not torch.isfinite(value).all()
            for value in (
                fold_predictions,
                fold_confidences,
                fold_corrections,
            )
        ):
            raise FloatingPointError(
                f"V9.8 fold {fold} candidate pool incomplete"
            )
        predictions[holdout_top] = fold_predictions
        confidences[holdout_top] = fold_confidences
        corrections[holdout_top] = fold_corrections
        metadata = {
            "outer_fold": fold,
            "inner_train_count": len(spec.inner_train_indices),
            "inner_valid_count": len(spec.inner_valid_indices),
            "development_count": len(development_top),
            "holdout_count": len(holdout_top),
            "inner_train_groups": list(spec.inner_train_groups),
            "inner_valid_groups": list(spec.inner_valid_groups),
            "holdout_groups": list(spec.outer_holdout_groups),
            "upstream_checkpoints": {
                name: str(path)
                for name, path in checkpoint_paths.items()
            },
            "checkpoint_sha256": checkpoint_sha,
            "teacher_candidates": list(TEACHER_NAMES),
            "teacher_anchor_index": int(teacher["anchor_index"]),
            "teacher_anchor_name": TEACHER_NAMES[
                int(teacher["anchor_index"])
            ],
            "valid_teacher_mae": teacher["valid_teacher_mae"],
            "global_teacher_weights": teacher["global_weights"].tolist(),
            "role_teacher_weights": teacher["role_weights"].tolist(),
            "selected_regularizations": teacher[
                "selected_regularizations"
            ],
            "region_counts": teacher["region_counts"],
        }
        fold_rows.append(metadata)
        torch.save(
            {
                "version": POOL_VERSION,
                "role_config": role_config.__dict__,
                "tail_config": tail_config.__dict__,
                "checkpoint_sha256": checkpoint_sha,
                "predictions": fold_predictions,
                "confidences": fold_confidences,
                "corrections": fold_corrections,
                "history": fold_history,
                "metadata": metadata,
            },
            fold_path,
        )
        logger.info(
            "V9.8 strict frontier outer fold=%d complete "
            "development=%d holdout=%d",
            fold,
            len(development_top),
            len(holdout_top),
        )

    if any(
        not torch.isfinite(value).all()
        for value in (predictions, confidences, corrections)
    ):
        raise FloatingPointError("V9.8 strict candidate pool is incomplete")
    actions = stack_action_predictions(top["oof_prediction"], predictions)
    targets = attainable_frontier_targets(actions, top["labels"])
    provenance = {
        "top_oof_cache": str(top_oof_cache_path),
        "is_fully_nested_teacher_stack": True,
        "holdout_label_isolation": True,
        "historical_full_train_teacher_reuse": False,
        "upstream_source": (
            "The clean DLF, ModDrop evaluator, and CFCompatKD checkpoint "
            "from the matching V9.2 outer fold. Each was trained and "
            "selected using only that fold's inner-train/inner-valid groups."
        ),
        "final_deployment_experts": (
            "Original frozen V9.3 expert checkpoints are used only on "
            "Validation/Test."
        ),
    }
    payload = {
        "version": POOL_VERSION,
        "method": (
            "strict_nested_v93_architecture_attainable_frontier_pool_v9_8"
        ),
        "sample_ids": top_ids,
        "group_ids": [conversation_group_id(value) for value in top_ids],
        "labels": top["labels"].float(),
        "anchor": top["oof_prediction"].float(),
        "function_space": top["oof_feature"].float(),
        "feature_space": top["feature_space"],
        "fold_index": top["fold_index"].long(),
        "expert_predictions": predictions,
        "expert_confidences": confidences,
        "expert_corrections": corrections,
        "action_names": ACTION_NAMES,
        "oracle_action_index": targets["oracle_index"],
        "oracle_value": targets["oracle_value"],
        "oracle_gain": targets["oracle_gain"],
        "oracle_cost_margin": targets["cost_margin"],
        "role_config": role_config.__dict__,
        "tail_config": tail_config.__dict__,
        "fold_metadata": fold_rows,
        "provenance": provenance,
    }
    torch.save(payload, final_path)
    pd.DataFrame(histories).to_csv(
        output_dir / "v98_strict_nested_specialist_history.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "inner_train_groups",
                    "inner_valid_groups",
                    "holdout_groups",
                    "upstream_checkpoints",
                    "checkpoint_sha256",
                    "global_teacher_weights",
                    "role_teacher_weights",
                    "selected_regularizations",
                    "region_counts",
                }
            }
            for row in fold_rows
        ]
    ).to_csv(
        output_dir / "v98_strict_nested_fold_summary.csv",
        index=False,
    )

    diagnostic = pd.DataFrame(
        {
            "sample_id": top_ids,
            "group_id": payload["group_ids"],
            "outer_fold": top["fold_index"].tolist(),
            "label": top["labels"].view(-1).tolist(),
            "anchor": top["oof_prediction"].view(-1).tolist(),
            "oracle_action": [
                ACTION_NAMES[index]
                for index in targets["oracle_index"].tolist()
            ],
            "oracle_value": targets["oracle_value"].view(-1).tolist(),
            "oracle_gain": targets["oracle_gain"].view(-1).tolist(),
            "oracle_cost_margin": targets["cost_margin"].view(-1).tolist(),
        }
    )
    for index, name in enumerate(SPECIALIST_NAMES):
        diagnostic[f"{name}_prediction"] = (
            predictions[:, index, 0].tolist()
        )
        diagnostic[f"{name}_confidence"] = (
            confidences[:, index, 0].tolist()
        )
    diagnostic.to_csv(
        output_dir / "v98_strict_nested_frontier_pool.csv",
        index=False,
    )

    labels = top["labels"].view(-1)
    expert_global_mae = {
        name: float(
            torch.abs(predictions[:, index, 0] - labels).mean().item()
        )
        for index, name in enumerate(SPECIALIST_NAMES)
    }
    summary = {
        "version": POOL_VERSION,
        "sample_count": n,
        "group_count": len(set(payload["group_ids"])),
        "fold_count": len(specs),
        "anchor_oof_mae": float(
            torch.abs(top["oof_prediction"] - top["labels"])
            .mean()
            .item()
        ),
        "sample_oracle_oof_mae": float(
            torch.abs(targets["oracle_value"] - top["labels"])
            .mean()
            .item()
        ),
        "mean_oracle_gain": float(targets["oracle_gain"].mean().item()),
        "meaningful_gain_rate": float(
            (targets["oracle_gain"].view(-1) > 0.02)
            .float()
            .mean()
            .item()
        ),
        "oracle_action_counts": {
            name: int(
                (targets["oracle_index"] == index).sum().item()
            )
            for index, name in enumerate(ACTION_NAMES)
        },
        "expert_global_mae": expert_global_mae,
        "fold_metadata": fold_rows,
        "provenance": provenance,
        "pool_path": str(final_path),
    }
    (
        output_dir / "v98_strict_nested_frontier_pool_summary.json"
    ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return payload
