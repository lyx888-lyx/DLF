"""Strict nested semantic shadow expert pool for V9.9 cost coaching."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch

from .frontier_expert_pool_v97 import (
    RoleCrossfitConfigV97,
    TailCrossfitConfigV97,
    _fit_role_expert,
    _fit_tail_expert,
)
from .model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    SPECIALIST_NAMES,
    actual_action_costs,
    stack_action_predictions,
)
from .oof_group_splits_v92 import build_subset_loader, conversation_group_id
from .semantic_expert_pool_v99 import (
    ROLE_INDEX,
    collect_role_semantic_rows,
    collect_tail_semantics,
)
from .strict_nested_frontier_pool_v98 import (
    FEATURE_SPACE_VERSION,
    TEACHER_NAMES,
    _build_fold_teacher_artifacts,
    _canonical_ids,
    _collect_fold_cfcompat_space,
    _dataset_ids,
    _fold_checkpoint_paths,
    _index_map,
    _reconstruct_outer_specs,
    _sha256,
)

logger = logging.getLogger("MMSA")
POOL_VERSION = "v99_strict_nested_semantic_expert_pool"


def build_strict_semantic_expert_pool_v99(
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
    """Build holdout-isolated candidate values and semantic signatures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "strict_semantic_expert_pool_v99.pth"
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu")
        if (
            payload.get("version") == POOL_VERSION
            and payload.get("signature_version") == SIGNATURE_VERSION
            and payload.get("role_config") == role_config.__dict__
            and payload.get("tail_config") == tail_config.__dict__
        ):
            logger.info("Reusing strict V9.9 semantic pool: %s", final_path)
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
        raise ValueError("V9.9 requires aligned V9.2 function-space features")

    specs, manifest = _reconstruct_outer_specs(top)
    manifest.to_csv(output_dir / "v99_strict_semantic_manifest.csv", index=False)
    top_ids = _canonical_ids(top["sample_ids"])
    top_id_map = _index_map(top_ids, "top-OOF")
    dataset_ids = _dataset_ids(train_dataset)
    dataset_id_map = _index_map(dataset_ids, "train-dataset")
    missing = sorted(set(top_ids) - set(dataset_id_map))
    if missing:
        raise RuntimeError(f"Train dataset misses {len(missing)} top OOF ids")
    top_to_dataset = [dataset_id_map[value] for value in top_ids]

    n = len(top_ids)
    signature_dim = len(SIGNATURE_FIELDS)
    predictions = torch.full((n, 4, 1), float("nan"))
    confidences = torch.full_like(predictions, float("nan"))
    corrections = torch.full_like(predictions, float("nan"))
    signatures = torch.full((n, 4, signature_dim), float("nan"))
    histories = []
    fold_rows = []
    name_to_column = {
        name: index for index, name in enumerate(SPECIALIST_NAMES)
    }

    for spec in specs:
        fold = int(spec.outer_fold)
        fold_dir = output_dir / f"outer_fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_path = fold_dir / "strict_semantic_holdout_v99.pth"
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
                and cached.get("signature_version") == SIGNATURE_VERSION
                and cached.get("role_config") == role_config.__dict__
                and cached.get("tail_config") == tail_config.__dict__
                and cached.get("checkpoint_sha256") == checkpoint_sha
            ):
                predictions[holdout_top] = cached["predictions"]
                confidences[holdout_top] = cached["confidences"]
                corrections[holdout_top] = cached["corrections"]
                signatures[holdout_top] = cached["signatures"]
                histories.extend(cached.get("history", []))
                fold_rows.append(cached["metadata"])
                logger.info("V9.9 semantic outer fold=%d resumed", fold)
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
        fold_signatures = torch.full(
            (len(holdout_top), 4, signature_dim), float("nan")
        )
        holdout_position = {
            int(top_index): offset
            for offset, top_index in enumerate(holdout_top.tolist())
        }
        fold_history = []

        holdout_loader = build_subset_loader(
            train_dataset,
            holdout_dataset_indices,
            role_config.batch_size,
            num_workers,
            False,
            0,
        )
        for offset, role in enumerate(("boundary", "positive")):
            role_seed = (
                int(args.seed)
                + 51001 * (fold + 1)
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
            rows = collect_role_semantic_rows(
                model,
                holdout_loader,
                args.device,
                ROLE_INDEX[role],
                top_id_map=top_id_map,
            )
            column = name_to_column[role]
            for row in rows:
                local = holdout_position[int(row["top_index"])]
                fold_predictions[local, column] = row["prediction"].view(1)
                fold_confidences[local, column] = row["confidence"].view(1)
                fold_corrections[local, column] = row["correction"].view(1)
                fold_signatures[local, column] = row["signature"]
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
                + 61001 * (fold + 1)
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
            collected = collect_tail_semantics(
                model,
                top["oof_feature"][holdout_top],
                top["oof_prediction"][holdout_top],
                args.device,
                tail_config.batch_size,
            )
            column = name_to_column[role]
            fold_predictions[:, column] = collected["prediction"]
            fold_confidences[:, column] = collected["confidence"]
            fold_corrections[:, column] = collected["correction"]
            fold_signatures[:, column] = collected["signature"]
            tagged = [{"outer_fold": fold, **row} for row in history]
            histories.extend(tagged)
            fold_history.extend(tagged)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        tensors = (
            fold_predictions,
            fold_confidences,
            fold_corrections,
            fold_signatures,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise FloatingPointError(
                f"V9.9 semantic fold {fold} candidate pool incomplete"
            )
        predictions[holdout_top] = fold_predictions
        confidences[holdout_top] = fold_confidences
        corrections[holdout_top] = fold_corrections
        signatures[holdout_top] = fold_signatures
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
                name: str(path) for name, path in checkpoint_paths.items()
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
                "signature_version": SIGNATURE_VERSION,
                "role_config": role_config.__dict__,
                "tail_config": tail_config.__dict__,
                "checkpoint_sha256": checkpoint_sha,
                "predictions": fold_predictions,
                "confidences": fold_confidences,
                "corrections": fold_corrections,
                "signatures": fold_signatures,
                "history": fold_history,
                "metadata": metadata,
            },
            fold_path,
        )
        logger.info(
            "V9.9 semantic outer fold=%d complete development=%d holdout=%d",
            fold,
            len(development_top),
            len(holdout_top),
        )

    tensors = (predictions, confidences, corrections, signatures)
    if any(not torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("V9.9 strict semantic pool is incomplete")

    actions = stack_action_predictions(top["oof_prediction"], predictions)
    costs = actual_action_costs(actions, top["labels"])
    oracle_index = costs.argmin(dim=1)
    oracle_cost = costs.gather(1, oracle_index.view(-1, 1))
    oracle_value = actions.squeeze(-1).gather(
        1, oracle_index.view(-1, 1)
    )
    sorted_cost = costs.sort(dim=1).values
    oracle_gain = costs[:, :1] - oracle_cost
    provenance = {
        "top_oof_cache": str(top_oof_cache_path),
        "is_fully_nested_teacher_stack": True,
        "holdout_label_isolation": True,
        "historical_full_train_teacher_reuse": False,
        "signature_version": SIGNATURE_VERSION,
        "semantic_fields": list(SIGNATURE_FIELDS),
        "upstream_source": (
            "Matching V9.2 outer-fold clean, ModDrop, and CFCompatKD "
            "checkpoints only; all exclude the complete outer holdout groups."
        ),
        "final_deployment_experts": (
            "Original frozen V9.3 checkpoints are used on Validation/Test and "
            "expose the same fixed semantic signature schema."
        ),
    }
    payload = {
        "version": POOL_VERSION,
        "method": "strict_nested_semantic_expert_pool_v9_9",
        "signature_version": SIGNATURE_VERSION,
        "signature_fields": list(SIGNATURE_FIELDS),
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
        "expert_signatures": signatures,
        "action_names": ACTION_NAMES,
        "action_costs": costs,
        "oracle_action_index": oracle_index,
        "oracle_value": oracle_value,
        "oracle_gain": oracle_gain,
        "oracle_cost_margin": sorted_cost[:, 1:2] - sorted_cost[:, :1],
        "role_config": role_config.__dict__,
        "tail_config": tail_config.__dict__,
        "fold_metadata": fold_rows,
        "provenance": provenance,
    }
    torch.save(payload, final_path)
    pd.DataFrame(histories).to_csv(
        output_dir / "v99_strict_semantic_specialist_history.csv",
        index=False,
    )

    diagnostic = pd.DataFrame(
        {
            "sample_id": top_ids,
            "group_id": payload["group_ids"],
            "outer_fold": top["fold_index"].tolist(),
            "label": top["labels"].view(-1).tolist(),
            "anchor": top["oof_prediction"].view(-1).tolist(),
            "oracle_action": [ACTION_NAMES[i] for i in oracle_index.tolist()],
            "oracle_value": oracle_value.view(-1).tolist(),
            "oracle_gain": oracle_gain.view(-1).tolist(),
            "oracle_cost_margin": payload["oracle_cost_margin"]
            .view(-1)
            .tolist(),
        }
    )
    for index, name in enumerate(SPECIALIST_NAMES):
        diagnostic[f"{name}_prediction"] = predictions[:, index, 0].tolist()
        diagnostic[f"{name}_confidence"] = confidences[:, index, 0].tolist()
        for field_index, field in enumerate(SIGNATURE_FIELDS):
            diagnostic[f"{name}_{field}"] = signatures[
                :, index, field_index
            ].tolist()
    diagnostic.to_csv(
        output_dir / "v99_strict_semantic_expert_pool.csv", index=False
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
        "signature_version": SIGNATURE_VERSION,
        "signature_dim": len(SIGNATURE_FIELDS),
        "sample_count": n,
        "group_count": len(set(payload["group_ids"])),
        "fold_count": len(specs),
        "anchor_oof_mae": float(costs[:, 0].mean().item()),
        "sample_oracle_oof_mae": float(oracle_cost.mean().item()),
        "mean_oracle_gain": float(oracle_gain.mean().item()),
        "meaningful_gain_rate": float(
            (oracle_gain.view(-1) > 0.02).float().mean().item()
        ),
        "oracle_action_counts": {
            name: int((oracle_index == index).sum().item())
            for index, name in enumerate(ACTION_NAMES)
        },
        "expert_global_mae": expert_global_mae,
        "fold_metadata": fold_rows,
        "provenance": provenance,
        "pool_path": str(final_path),
    }
    (
        output_dir / "v99_strict_semantic_expert_pool_summary.json"
    ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return payload
