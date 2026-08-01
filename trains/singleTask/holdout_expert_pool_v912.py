"""Build one holdout-isolated expert stack and reuse it on Router/Valid/Test."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import torch

from .frontier_expert_pool_v97 import (
    RoleCrossfitConfigV97,
    TailCrossfitConfigV97,
    _fit_role_expert,
    _fit_tail_expert,
)
from .model.SemanticCostCoachV99 import (
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    SPECIALIST_NAMES,
)
from .oof_group_splits_v92 import (
    build_subset_loader,
    canonical_sample_id,
    conversation_group_id,
)
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
    _collect_teacher_predictions,
    _dataset_ids,
    _fold_checkpoint_paths,
    _index_map,
    _reconstruct_outer_specs,
    _sha256,
)

logger = logging.getLogger("MMSA")
POOL_VERSION = "v912_single_holdout_same_expert_stack"


def _cpu_state(model) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _all_indices(dataset) -> list[int]:
    return list(range(len(dataset)))


def _role_values(
    rows,
    ordered_sample_ids: Sequence[str],
) -> Dict[str, torch.Tensor]:
    by_id = {}
    for row in rows:
        sample_id = canonical_sample_id(row["sample_id"])
        if sample_id in by_id:
            raise RuntimeError(f"duplicate role semantic sample id: {sample_id}")
        by_id[sample_id] = row
    expected = [canonical_sample_id(value) for value in ordered_sample_ids]
    missing = [sample_id for sample_id in expected if sample_id not in by_id]
    if missing:
        raise RuntimeError(
            f"role semantic collection misses {len(missing)} samples"
        )
    ordered = [by_id[sample_id] for sample_id in expected]
    return {
        "prediction": torch.cat(
            [row["prediction"] for row in ordered], dim=0
        ).view(-1, 1),
        "confidence": torch.cat(
            [row["confidence"] for row in ordered], dim=0
        ).view(-1, 1),
        "correction": torch.cat(
            [row["correction"] for row in ordered], dim=0
        ).view(-1, 1),
        "signature": torch.stack(
            [row["signature"] for row in ordered], dim=0
        ),
    }


def _collect_split(
    args,
    dataset,
    indices: Sequence[int],
    checkpoint_paths: Mapping[str, Path],
    teacher,
    role_models,
    tail_models,
    role_config: RoleCrossfitConfigV97,
    tail_config: TailCrossfitConfigV97,
    num_workers: int,
):
    indices = [int(value) for value in indices]
    teacher_data = _collect_teacher_predictions(
        args,
        dataset,
        indices,
        checkpoint_paths,
        role_config.batch_size,
        num_workers,
    )
    function_data = _collect_fold_cfcompat_space(
        args,
        dataset,
        indices,
        checkpoint_paths["cfcompat"],
        tail_config.batch_size,
        num_workers,
    )
    if teacher_data["sample_ids"] != function_data["sample_ids"]:
        raise RuntimeError("teacher/function-space sample order mismatch")
    if not torch.allclose(
        teacher_data["labels"].float(),
        function_data["labels"].float(),
    ):
        raise RuntimeError("teacher/function-space labels mismatch")

    anchor = teacher_data["predictions"][
        :, int(teacher["anchor_index"])
    ].float()
    loader = build_subset_loader(
        dataset,
        indices,
        role_config.batch_size,
        num_workers,
        False,
        0,
    )
    experts: Dict[str, Dict[str, object]] = {}
    for name in ("boundary", "positive"):
        rows = collect_role_semantic_rows(
            role_models[name],
            loader,
            args.device,
            ROLE_INDEX[name],
        )
        experts[name] = _role_values(rows, teacher_data["sample_ids"])

    for name in ("strong_negative", "strong_positive"):
        values = collect_tail_semantics(
            tail_models[name],
            function_data["feature"],
            anchor,
            args.device,
            tail_config.batch_size,
        )
        experts[name] = {
            "prediction": values["prediction"].float(),
            "confidence": values["confidence"].float(),
            "correction": values["correction"].float(),
            "signature": values["signature"].float(),
        }

    expected_shape = anchor.shape
    for name in SPECIALIST_NAMES:
        values = experts[name]
        if values["prediction"].shape != expected_shape:
            raise RuntimeError(
                f"{name} prediction shape mismatch: "
                f"{tuple(values['prediction'].shape)} vs {tuple(expected_shape)}"
            )
        if values["signature"].shape[1] != len(SIGNATURE_FIELDS):
            raise RuntimeError(f"{name} semantic signature width mismatch")
        if not torch.isfinite(values["signature"]).all():
            raise FloatingPointError(f"{name} semantic signature is non-finite")

    return {
        "sample_ids": list(teacher_data["sample_ids"]),
        "group_ids": [
            conversation_group_id(value)
            for value in teacher_data["sample_ids"]
        ],
        "labels": teacher_data["labels"].float(),
        "anchor": anchor,
        "function_space": function_data["feature"].float(),
        "feature_space": FEATURE_SPACE_VERSION,
        "signature_version": SIGNATURE_VERSION,
        "signature_fields": list(SIGNATURE_FIELDS),
        "experts": experts,
    }


def build_holdout_expert_pool_v912(
    args,
    datasets,
    top_oof_cache_path: Path,
    output_dir: Path,
    outer_fold: int,
    role_config: RoleCrossfitConfigV97,
    tail_config: TailCrossfitConfigV97,
    num_workers: int = 1,
    teacher_fit_steps: int = 300,
    min_region_samples: int = 10,
    resume: bool = True,
):
    """Train one expert stack without Router holdout labels and reuse it everywhere."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "holdout_expert_pool_v912.pth"

    top_oof_cache_path = Path(top_oof_cache_path)
    top = torch.load(top_oof_cache_path, map_location="cpu")
    required = {
        "sample_ids",
        "labels",
        "fold_index",
        "feature_space",
        "outer_folds",
        "inner_valid_fraction",
        "seed",
    }
    if not required.issubset(top):
        raise ValueError(
            f"V9.12 top OOF cache missing {sorted(required - set(top))}"
        )
    if top.get("feature_space") != FEATURE_SPACE_VERSION:
        raise ValueError("V9.12 requires the aligned V9.2 function space")

    specs, manifest = _reconstruct_outer_specs(top)
    selected = None
    for spec in specs:
        if int(spec.outer_fold) == int(outer_fold):
            selected = spec
            break
    if selected is None:
        raise ValueError(
            f"outer_fold={outer_fold} is unavailable; "
            f"choices={[int(value.outer_fold) for value in specs]}"
        )

    checkpoint_paths = _fold_checkpoint_paths(
        top_oof_cache_path, int(outer_fold)
    )
    checkpoint_sha = {
        name: _sha256(path) for name, path in checkpoint_paths.items()
    }
    resume_key = {
        "version": POOL_VERSION,
        "outer_fold": int(outer_fold),
        "role_config": role_config.__dict__,
        "tail_config": tail_config.__dict__,
        "teacher_fit_steps": int(teacher_fit_steps),
        "min_region_samples": int(min_region_samples),
        "checkpoint_sha256": checkpoint_sha,
    }
    if resume and final_path.is_file():
        payload = torch.load(final_path, map_location="cpu")
        if all(payload.get(key) == value for key, value in resume_key.items()):
            logger.info("Reusing V9.12 holdout expert pool: %s", final_path)
            return payload

    train_dataset = datasets["train"]
    top_ids = _canonical_ids(top["sample_ids"])
    dataset_ids = _dataset_ids(train_dataset)
    dataset_map = _index_map(dataset_ids, "train-dataset")
    missing = sorted(set(top_ids) - set(dataset_map))
    if missing:
        raise RuntimeError(
            f"Train dataset misses {len(missing)} top-cache sample ids"
        )
    top_to_dataset = [dataset_map[value] for value in top_ids]

    development_top = sorted(
        set(selected.inner_train_indices) | set(selected.inner_valid_indices)
    )
    router_top = list(selected.outer_holdout_indices)
    development_indices = [
        top_to_dataset[int(index)] for index in development_top
    ]
    inner_valid_indices = [
        top_to_dataset[int(index)] for index in selected.inner_valid_indices
    ]
    router_indices = [
        top_to_dataset[int(index)] for index in router_top
    ]

    development_groups = {
        conversation_group_id(top_ids[int(index)])
        for index in development_top
    }
    router_groups = {
        conversation_group_id(top_ids[int(index)])
        for index in router_top
    }
    overlap = development_groups & router_groups
    if overlap:
        raise RuntimeError(
            f"V9.12 development/router group overlap: {sorted(overlap)[:5]}"
        )

    split_rows = []
    for top_index in development_top:
        split_rows.append(
            {
                "sample_id": top_ids[int(top_index)],
                "group_id": conversation_group_id(top_ids[int(top_index)]),
                "split": "expert_train",
            }
        )
    for top_index in router_top:
        split_rows.append(
            {
                "sample_id": top_ids[int(top_index)],
                "group_id": conversation_group_id(top_ids[int(top_index)]),
                "split": "router_train",
            }
        )
    pd.DataFrame(split_rows).to_csv(
        output_dir / "v912_holdout_split_manifest.csv",
        index=False,
    )
    manifest.to_csv(
        output_dir / "v912_source_outer_fold_manifest.csv",
        index=False,
    )

    teacher = _build_fold_teacher_artifacts(
        args,
        train_dataset,
        development_indices,
        inner_valid_indices,
        checkpoint_paths,
        role_config.batch_size,
        num_workers,
        teacher_fit_steps,
        min_region_samples,
    )
    development_space = _collect_fold_cfcompat_space(
        args,
        train_dataset,
        development_indices,
        checkpoint_paths["cfcompat"],
        tail_config.batch_size,
        num_workers,
    )
    expected_development_ids = [
        top_ids[int(index)] for index in development_top
    ]
    if development_space["sample_ids"] != expected_development_ids:
        raise RuntimeError("development function-space order is misaligned")
    teacher_train = teacher["cache"]["splits"]["train"]
    if not torch.allclose(
        development_space["labels"].float(),
        teacher_train["labels"].float(),
    ):
        raise RuntimeError("development teacher/function-space labels mismatch")
    selected_anchor = teacher_train["predictions"][
        :, int(teacher["anchor_index"])
    ].float()
    if len(selected_anchor) != len(development_space["feature"]):
        raise RuntimeError("development anchor/function-space length mismatch")

    histories = []
    role_models = {}
    role_checkpoint_paths = {}
    for offset, name in enumerate(("boundary", "positive")):
        seed = int(args.seed) + 912101 + 101 * (offset + 1)
        model, history = _fit_role_expert(
            args,
            train_dataset,
            development_indices,
            name,
            teacher,
            role_config,
            seed,
            num_workers,
        )
        histories.extend(history)
        checkpoint = output_dir / f"{name}_holdout_expert_v912.pth"
        torch.save(
            {
                "method": POOL_VERSION,
                "role": name,
                "outer_fold": int(outer_fold),
                "state_dict": _cpu_state(model),
                "role_config": role_config.__dict__,
                "upstream_checkpoint_sha256": checkpoint_sha,
            },
            checkpoint,
        )
        role_models[name] = model
        role_checkpoint_paths[name] = checkpoint

    tail_models = {}
    tail_checkpoint_paths = {}
    for offset, name in enumerate(
        ("strong_negative", "strong_positive")
    ):
        seed = int(args.seed) + 912503 + 131 * (offset + 1)
        model, history = _fit_tail_expert(
            development_space["feature"],
            selected_anchor,
            development_space["labels"],
            name,
            tail_config,
            args.device,
            seed,
        )
        histories.extend(history)
        checkpoint = output_dir / f"{name}_holdout_expert_v912.pth"
        torch.save(
            {
                "method": POOL_VERSION,
                "role": name,
                "outer_fold": int(outer_fold),
                "state_dict": _cpu_state(model),
                "tail_config": tail_config.__dict__,
                "upstream_checkpoint_sha256": checkpoint_sha,
            },
            checkpoint,
        )
        tail_models[name] = model
        tail_checkpoint_paths[name] = checkpoint

    split_pools = {
        "router_train": _collect_split(
            args,
            train_dataset,
            router_indices,
            checkpoint_paths,
            teacher,
            role_models,
            tail_models,
            role_config,
            tail_config,
            num_workers,
        ),
        "valid": _collect_split(
            args,
            datasets["valid"],
            _all_indices(datasets["valid"]),
            checkpoint_paths,
            teacher,
            role_models,
            tail_models,
            role_config,
            tail_config,
            num_workers,
        ),
        "test": _collect_split(
            args,
            datasets["test"],
            _all_indices(datasets["test"]),
            checkpoint_paths,
            teacher,
            role_models,
            tail_models,
            role_config,
            tail_config,
            num_workers,
        ),
    }

    pd.DataFrame(histories).to_csv(
        output_dir / "v912_holdout_expert_training_history.csv",
        index=False,
    )
    payload = {
        **resume_key,
        "method": "single_holdout_same_expert_stack_v9_12",
        "top_oof_cache": str(top_oof_cache_path),
        "source_outer_fold_count": int(top["outer_folds"]),
        "expert_train_count": len(development_indices),
        "router_train_count": len(router_indices),
        "expert_train_groups": sorted(development_groups),
        "router_train_groups": sorted(router_groups),
        "inner_valid_groups": list(selected.inner_valid_groups),
        "teacher_names": list(TEACHER_NAMES),
        "teacher_anchor_index": int(teacher["anchor_index"]),
        "teacher_anchor_name": TEACHER_NAMES[int(teacher["anchor_index"])],
        "valid_teacher_mae": dict(teacher["valid_teacher_mae"]),
        "upstream_checkpoints": {
            name: str(path) for name, path in checkpoint_paths.items()
        },
        "role_expert_checkpoints": {
            name: str(path) for name, path in role_checkpoint_paths.items()
        },
        "tail_expert_checkpoints": {
            name: str(path) for name, path in tail_checkpoint_paths.items()
        },
        "splits": split_pools,
        "provenance": {
            "router_labels_used_for_expert_training": False,
            "router_groups_disjoint_from_expert_train": True,
            "same_expert_models_router_valid_test": True,
            "official_validation_used_for_expert_training": False,
            "official_test_used_for_training_or_selection": False,
            "deep_nested_coach_oof": False,
            "protocol": (
                "One fixed V9.2 outer holdout is Router-Train. "
                "One fold-local expert stack is trained on its development "
                "groups and reused unchanged on Router-Train, Validation, Test."
            ),
        },
    }
    torch.save(payload, final_path)

    for model in [*role_models.values(), *tail_models.values()]:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(
        "V9.12 expert pool complete fold=%d expert_train=%d router_train=%d",
        int(outer_fold),
        len(development_indices),
        len(router_indices),
    )
    return payload
