"""Fully same-stack expert construction for V9.19 nested cross-fitting.

Every call trains the same deployment recipe:
  1. clean DLF, ModDrop evaluator, and CFCompatKD anchor on train/valid groups;
  2. Boundary and Positive RoleConditionedDLF specialists using those fold-local
     teacher checkpoints and fold-local teacher predictions;
  3. Strong-Negative and Strong-Positive CachedTailResidualHead specialists on
     the same fold-local CFCompat function-space representation;
  4. collect one aligned Anchor + four-specialist pool on an unseen target set.

The recipe is used unchanged for inner OOF stacks and outer deployment stacks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import pandas as pd
import torch

from .cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _checkpoint_payload,
    _load_state,
    predict_plain,
    predict_wrapper_lav,
    train_cfcompat_student,
    train_clean_dlf,
    train_moddrop_evaluator,
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
    FEATURE_KEYS,
    FEATURE_SPACE_VERSION,
    predict_wrapper_function_space,
)
from .model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES
from .oof_group_splits_v92 import (
    StageLimits,
    build_subset_loader,
    canonical_sample_id,
    conversation_group_id,
)
from .role_conditioned_experts_v9 import fit_role_teacher_weights

logger = logging.getLogger("MMSA")
STACK_VERSION = "full_same_stack_expert_factory_v919_v1"
TEACHER_NAMES = ("clean", "moddrop", "cfcompat")
ROLE_NAMES = ("boundary", "positive")
TAIL_NAMES = ("strong_negative", "strong_positive")


@dataclass(frozen=True)
class SameStackConfigV919:
    feature_batch_size: int = 32
    num_workers: int = 1
    teacher_fit_steps: int = 300
    min_region_samples: int = 8
    force_cfcompat_anchor: bool = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_ids(dataset) -> list[str]:
    if not hasattr(dataset, "ids"):
        raise AttributeError("MMDataset must expose ids for V9.19")
    return [canonical_sample_id(value) for value in list(dataset.ids)]


def _ordered_unique(indices: Iterable[int]) -> list[int]:
    return sorted(set(int(value) for value in indices))


def _row_map(
    rows: Sequence[Mapping[str, object]], name: str
) -> Dict[int, Mapping[str, object]]:
    result: Dict[int, Mapping[str, object]] = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in result:
            raise RuntimeError(
                f"duplicate {name} row for sample index {index}"
            )
        result[index] = row
    return result


def _collect_teacher_rows(model, loader, device, wrapper: bool):
    return (
        predict_wrapper_lav(model, loader, device, False)
        if wrapper
        else predict_plain(model, loader, device, False)
    )


def _load_base_models(args, checkpoints: Mapping[str, Path]):
    clean = DLF(args).to(args.device)
    clean.load_state_dict(
        _load_state(checkpoints["clean"], args.device), strict=True
    )
    models = {"clean": clean}
    for name in ("moddrop", "cfcompat"):
        model = MissingModalityWrapper(
            DLF(args).to(args.device),
            int(args.feature_dims[1]),
            int(args.feature_dims[2]),
        ).to(args.device)
        model.load_state_dict(
            _load_state(checkpoints[name], args.device), strict=True
        )
        models[name] = model
    for model in models.values():
        model.eval()
    return models


def _collect_teacher_cache(
    args,
    dataset,
    indices: Sequence[int],
    models: Mapping[str, torch.nn.Module],
    batch_size: int,
    num_workers: int,
):
    ordered = _ordered_unique(indices)
    loader = build_subset_loader(
        dataset, ordered, batch_size, num_workers, False, 0
    )
    predictions = []
    sample_ids = None
    labels = None
    for name in TEACHER_NAMES:
        rows = _collect_teacher_rows(
            models[name], loader, args.device, wrapper=(name != "clean")
        )
        mapping = _row_map(rows, f"{name}-teacher")
        if set(mapping) != set(ordered):
            raise RuntimeError(
                f"{name} teacher rows differ from requested indices"
            )
        local_ids = [
            canonical_sample_id(mapping[index]["sample_id"])
            for index in ordered
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
        elif sample_ids != local_ids or not torch.allclose(
            labels, local_labels
        ):
            raise RuntimeError(
                "teacher sample alignment changed across candidates"
            )
        predictions.append(local_predictions)
    return {
        "indices": ordered,
        "sample_ids": sample_ids,
        "labels": labels,
        "predictions": torch.stack(predictions, dim=1),
    }


def _collect_function_space(
    args,
    dataset,
    indices: Sequence[int],
    cfcompat_model,
    batch_size: int,
    num_workers: int,
):
    ordered = _ordered_unique(indices)
    loader = build_subset_loader(
        dataset, ordered, batch_size, num_workers, False, 0
    )
    rows = predict_wrapper_function_space(
        cfcompat_model, loader, args.device, capture_features=True
    )
    mapping = _row_map(rows, "cfcompat-function-space")
    if set(mapping) != set(ordered):
        raise RuntimeError(
            "function-space rows differ from requested indices"
        )
    return {
        "indices": ordered,
        "sample_ids": [
            canonical_sample_id(mapping[index]["sample_id"])
            for index in ordered
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
            [
                mapping[index]["feature"].float().view(-1)
                for index in ordered
            ],
            dim=0,
        ),
    }


def _checkpoint_metadata(checkpoints: Mapping[str, Path]):
    return {
        name: {
            **_checkpoint_payload(path),
            "sha256": sha256(path),
        }
        for name, path in checkpoints.items()
    }


def _pool_is_reusable(
    path: Path,
    recipe: Mapping[str, object],
    target_indices: Sequence[int],
) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu")
    return (
        payload.get("version") == STACK_VERSION
        and payload.get("recipe") == dict(recipe)
        and payload.get("sample_indices")
        == [int(value) for value in target_indices]
    )


def train_and_collect_same_stack(
    args,
    dataset,
    train_indices: Sequence[int],
    valid_indices: Sequence[int],
    target_indices: Sequence[int],
    output_dir: Path,
    stage_limits: StageLimits,
    role_config: RoleCrossfitConfigV97,
    tail_config: TailCrossfitConfigV97,
    stack_config: SameStackConfigV919,
    seed: int,
    resume: bool = True,
):
    """Train one complete fold-local stack and predict an unseen target set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "same_stack_target_pool_v919.pth"
    train_indices = _ordered_unique(train_indices)
    valid_indices = _ordered_unique(valid_indices)
    target_indices = _ordered_unique(target_indices)
    fit_indices = _ordered_unique([*train_indices, *valid_indices])
    if set(train_indices) & set(valid_indices):
        raise RuntimeError("same-stack train/valid overlap")
    if set(fit_indices) & set(target_indices):
        raise RuntimeError("same-stack target entered stack fitting")

    recipe = {
        "stage_limits": asdict(stage_limits),
        "role_config": asdict(role_config),
        "tail_config": asdict(tail_config),
        "stack_config": asdict(stack_config),
        "seed": int(seed),
        "train_indices": train_indices,
        "valid_indices": valid_indices,
    }
    if resume and _pool_is_reusable(
        pool_path, recipe, target_indices
    ):
        logger.info("Reusing V9.19 stack pool: %s", pool_path)
        return torch.load(pool_path, map_location="cpu")

    train_loader = build_subset_loader(
        dataset,
        train_indices,
        int(args.batch_size),
        stack_config.num_workers,
        True,
        seed,
    )
    valid_loader = build_subset_loader(
        dataset,
        valid_indices,
        int(args.batch_size),
        stack_config.num_workers,
        False,
        seed,
    )
    checkpoints = {
        "clean": output_dir / "clean_dlf_best_inner_valid.pth",
        "moddrop": output_dir / "moddrop_evaluator_best_inner_valid.pth",
        "cfcompat": output_dir / "cfcompat_student_best_inner_valid.pth",
    }

    def run_or_resume_stage(name, checkpoint, trainer):
        summary_path = output_dir / f"{name}_stage_summary.json"
        history_path = output_dir / f"{name}_history.csv"
        if resume and checkpoint.is_file() and summary_path.is_file():
            logger.info("Reusing V9.19 %s stage: %s", name, checkpoint)
            return json.loads(summary_path.read_text(encoding="utf-8"))
        summary = trainer()
        history = summary.get("history", [])
        pd.DataFrame(history).to_csv(history_path, index=False)
        compact = {
            key: value for key, value in summary.items() if key != "history"
        }
        summary_path.write_text(
            json.dumps(compact, indent=2), encoding="utf-8"
        )
        return compact

    clean_summary = run_or_resume_stage(
        "clean",
        checkpoints["clean"],
        lambda: train_clean_dlf(
            args,
            train_loader,
            valid_loader,
            checkpoints["clean"],
            stage_limits,
        ),
    )
    moddrop_summary = run_or_resume_stage(
        "moddrop",
        checkpoints["moddrop"],
        lambda: train_moddrop_evaluator(
            args,
            train_loader,
            valid_loader,
            checkpoints["clean"],
            checkpoints["moddrop"],
            stage_limits,
            seed,
        ),
    )
    cfcompat_summary = run_or_resume_stage(
        "cfcompat",
        checkpoints["cfcompat"],
        lambda: train_cfcompat_student(
            args,
            train_loader,
            valid_loader,
            checkpoints["clean"],
            checkpoints["moddrop"],
            checkpoints["cfcompat"],
            stage_limits,
            seed,
        ),
    )

    base_models = _load_base_models(args, checkpoints)
    fit_teacher = _collect_teacher_cache(
        args,
        dataset,
        fit_indices,
        base_models,
        stack_config.feature_batch_size,
        stack_config.num_workers,
    )
    valid_teacher = _collect_teacher_cache(
        args,
        dataset,
        valid_indices,
        base_models,
        stack_config.feature_batch_size,
        stack_config.num_workers,
    )
    valid_mae = torch.abs(
        valid_teacher["predictions"]
        - valid_teacher["labels"].unsqueeze(1)
    ).mean(dim=(0, 2))
    anchor_index = (
        2
        if stack_config.force_cfcompat_anchor
        else int(valid_mae.argmin().item())
    )
    teacher_fit = fit_role_teacher_weights(
        valid_teacher["predictions"],
        valid_teacher["labels"],
        valid_teacher["sample_ids"],
        steps=stack_config.teacher_fit_steps,
        min_region_samples=stack_config.min_region_samples,
    )
    pd.DataFrame(teacher_fit["cv_rows"]).to_csv(
        output_dir / "role_teacher_cv.csv", index=False
    )
    teacher = {
        "cache": {
            "splits": {
                "train": {
                    "sample_ids": fit_teacher["sample_ids"],
                    "labels": fit_teacher["labels"],
                    "predictions": fit_teacher["predictions"],
                }
            }
        },
        "teacher_paths": [checkpoints[name] for name in TEACHER_NAMES],
        "anchor_index": int(anchor_index),
        "global_weights": teacher_fit["global_weights"].float(),
        "role_weights": teacher_fit["role_weights"].float(),
    }

    dataset_ids = _dataset_ids(dataset)
    target_id_map = {
        dataset_ids[index]: index for index in target_indices
    }
    role_outputs: Dict[str, Dict[int, Mapping[str, object]]] = {}
    role_histories = []
    for offset, role in enumerate(ROLE_NAMES):
        role_seed = int(seed) + 10103 * (offset + 1)
        model, history = _fit_role_expert(
            args,
            dataset,
            fit_indices,
            role,
            teacher,
            role_config,
            role_seed,
            stack_config.num_workers,
        )
        rows = _collect_role_expert(
            model,
            args,
            dataset,
            target_indices,
            role,
            target_id_map,
            role_config.batch_size,
            stack_config.num_workers,
        )
        role_outputs[role] = {
            int(row["top_index"]): row for row in rows
        }
        role_histories.extend(
            {"expert": role, **row} for row in history
        )
        torch.save(
            {
                "version": STACK_VERSION,
                "role": role,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "role_config": asdict(role_config),
                "anchor_checkpoint": str(
                    checkpoints[TEACHER_NAMES[anchor_index]]
                ),
            },
            output_dir / f"{role}_expert_v919.pth",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(role_histories).to_csv(
        output_dir / "role_expert_history.csv", index=False
    )

    fit_space = _collect_function_space(
        args,
        dataset,
        fit_indices,
        base_models["cfcompat"],
        stack_config.feature_batch_size,
        stack_config.num_workers,
    )
    target_space = _collect_function_space(
        args,
        dataset,
        target_indices,
        base_models["cfcompat"],
        stack_config.feature_batch_size,
        stack_config.num_workers,
    )
    tail_outputs = {}
    tail_histories = []
    for offset, role in enumerate(TAIL_NAMES):
        tail_seed = int(seed) + 20107 * (offset + 1)
        model, history = _fit_tail_expert(
            fit_space["feature"],
            fit_space["anchor"],
            fit_space["labels"],
            role,
            tail_config,
            args.device,
            tail_seed,
        )
        tail_outputs[role] = _collect_tail_expert(
            model,
            target_space["feature"],
            target_space["anchor"],
            args.device,
            tail_config.batch_size,
        )
        tail_histories.extend(
            {"expert": role, **row} for row in history
        )
        torch.save(
            {
                "version": STACK_VERSION,
                "role": role,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "tail_config": asdict(tail_config),
            },
            output_dir / f"{role}_expert_v919.pth",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(tail_histories).to_csv(
        output_dir / "tail_expert_history.csv", index=False
    )

    n = len(target_indices)
    expert_predictions = torch.full((n, 4, 1), float("nan"))
    expert_confidences = torch.full_like(
        expert_predictions, float("nan")
    )
    expert_corrections = torch.full_like(
        expert_predictions, float("nan")
    )
    name_to_column = {
        name: index for index, name in enumerate(SPECIALIST_NAMES)
    }
    target_position = {
        index: position for position, index in enumerate(target_indices)
    }
    for role in ROLE_NAMES:
        column = name_to_column[role]
        for global_index, row in role_outputs[role].items():
            position = target_position[global_index]
            expert_predictions[position, column] = row[
                "prediction"
            ].view(1)
            expert_confidences[position, column] = row[
                "confidence"
            ].view(1)
            expert_corrections[position, column] = row[
                "correction"
            ].view(1)
    for role in TAIL_NAMES:
        column = name_to_column[role]
        expert_predictions[:, column] = tail_outputs[role]["prediction"]
        expert_confidences[:, column] = tail_outputs[role][
            "applicability_prob"
        ]
        expert_corrections[:, column] = tail_outputs[role]["correction"]

    if any(
        not torch.isfinite(value).all()
        for value in (
            expert_predictions,
            expert_confidences,
            expert_corrections,
        )
    ):
        raise FloatingPointError(
            "same-stack expert pool is incomplete"
        )
    if target_space["sample_ids"] != [
        dataset_ids[index] for index in target_indices
    ]:
        raise RuntimeError("same-stack target order changed")

    stage_summary = {
        "clean": clean_summary,
        "moddrop": moddrop_summary,
        "cfcompat": cfcompat_summary,
    }
    payload = {
        "version": STACK_VERSION,
        "method": "complete_fold_local_anchor_and_four_specialists",
        "recipe": recipe,
        "sample_indices": target_indices,
        "sample_ids": target_space["sample_ids"],
        "group_ids": [
            conversation_group_id(value)
            for value in target_space["sample_ids"]
        ],
        "labels": target_space["labels"].float(),
        "anchor": target_space["anchor"].float(),
        "function_space": target_space["feature"].float(),
        "feature_space": FEATURE_SPACE_VERSION,
        "feature_keys": list(FEATURE_KEYS),
        "expert_predictions": expert_predictions,
        "expert_confidences": expert_confidences,
        "expert_corrections": expert_corrections,
        "action_names": ACTION_NAMES,
        "teacher_names": TEACHER_NAMES,
        "teacher_valid_mae": {
            name: float(valid_mae[index].item())
            for index, name in enumerate(TEACHER_NAMES)
        },
        "teacher_anchor_name": TEACHER_NAMES[anchor_index],
        "teacher_global_weights": teacher_fit[
            "global_weights"
        ].tolist(),
        "teacher_role_weights": teacher_fit["role_weights"].tolist(),
        "stage_summary": stage_summary,
        "checkpoints": _checkpoint_metadata(checkpoints),
        "provenance": {
            "same_stack_recipe_for_inner_and_outer": True,
            "target_groups_excluded_from_anchor_training": True,
            "target_groups_excluded_from_expert_training": True,
            "target_groups_excluded_from_checkpoint_selection": True,
            "cfcompat_is_deployment_anchor": bool(
                stack_config.force_cfcompat_anchor
            ),
            "official_validation_or_test_used": False,
        },
    }
    torch.save(payload, pool_path)
    (output_dir / "same_stack_metadata_v919.json").write_text(
        json.dumps(
            {
                "version": STACK_VERSION,
                "sample_count": n,
                "train_count": len(train_indices),
                "valid_count": len(valid_indices),
                "target_count": len(target_indices),
                "teacher_anchor_name": payload["teacher_anchor_name"],
                "teacher_valid_mae": payload["teacher_valid_mae"],
                "stage_summary": stage_summary,
                "checkpoint_sha256": {
                    name: value["sha256"]
                    for name, value in payload["checkpoints"].items()
                },
                "pool_path": str(pool_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del base_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


__all__ = [
    "STACK_VERSION",
    "SameStackConfigV919",
    "train_and_collect_same_stack",
    "sha256",
]
