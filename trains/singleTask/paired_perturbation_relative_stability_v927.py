"""Paired model-input perturbation audit for V9.27.

V9.27 never trains or changes the frozen V9.19 anchor/specialists. It reruns
those exact checkpoints on deterministic, mild audio/vision feature-sequence
perturbations. The same perturbed action matrix is passed through the frozen
V9.21 convex-shrinkage weights, so every specialist is compared with the
baseline under exactly the same perturbation.

The repository uses pre-extracted audio and vision feature sequences. These
are model-input perturbations, not waveform/pixel edits.
"""

from __future__ import annotations

import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from .cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _load_state,
    mode_to_mask,
)
from .expert_self_risk_v925 import EXPERT_ACTION_INDEX, EXPERT_NAMES
from .frontier_expert_pool_v97 import ROLE_INDEX
from .function_space_features_v92 import function_space_feature
from .model.CachedTailResidualHeadV92 import CachedTailResidualHeadV92
from .model.RoleConditionedDLF import RoleConditionedDLF
from .model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES
from .no_train_decomposition_v920 import normalize_pool
from .oof_group_splits_v92 import build_subset_loader, canonical_sample_id
from .same_stack_expert_factory_v919 import sha256

AUDIT_VERSION = "paired_perturbation_relative_stability_v927_v1"
PRIMARY_METHOD = "paired_stability_plus_v925"
STABILITY_ONLY_METHOD = "paired_stability_only"
BASE_METHOD = "v925_base"


@dataclass(frozen=True)
class PerturbationSpecV927:
    name: str
    modality: str
    kind: str
    magnitude: float = 0.0
    location: float = 0.0
    seed_offset: int = 0


@dataclass(frozen=True)
class PairedPerturbationConfigV927:
    batch_size: int = 32
    num_workers: int = 1
    noise_scale: float = 0.01
    temporal_mask_fraction: float = 0.05
    identity_tolerance: float = 5e-4
    sign_epsilon: float = 1e-6
    ratio_epsilon: float = 1e-4
    base_seed: int = 1111

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if not 0.0 < self.noise_scale <= 0.10:
            raise ValueError("noise_scale must be in (0, 0.10]")
        if not 0.0 < self.temporal_mask_fraction <= 0.20:
            raise ValueError("temporal_mask_fraction must be in (0, 0.20]")
        if self.identity_tolerance <= 0.0:
            raise ValueError("identity_tolerance must be positive")


def perturbation_specs(config: PairedPerturbationConfigV927):
    """Return the fixed pre-registered identity + 12 perturbation set."""

    config.validate()
    n = float(config.noise_scale)
    m = float(config.temporal_mask_fraction)
    return (
        PerturbationSpecV927("identity", "none", "identity"),
        PerturbationSpecV927("audio_gain_095", "audio", "gain", 0.95),
        PerturbationSpecV927("audio_gain_105", "audio", "gain", 1.05),
        PerturbationSpecV927(
            "audio_noise_a", "audio", "noise", n, seed_offset=101
        ),
        PerturbationSpecV927(
            "audio_noise_b", "audio", "noise", n, seed_offset=211
        ),
        PerturbationSpecV927(
            "audio_mask_025", "audio", "mask", m, location=0.25
        ),
        PerturbationSpecV927(
            "audio_mask_065", "audio", "mask", m, location=0.65
        ),
        PerturbationSpecV927("vision_gain_095", "vision", "gain", 0.95),
        PerturbationSpecV927("vision_gain_105", "vision", "gain", 1.05),
        PerturbationSpecV927(
            "vision_noise_a", "vision", "noise", n, seed_offset=307
        ),
        PerturbationSpecV927(
            "vision_noise_b", "vision", "noise", n, seed_offset=419
        ),
        PerturbationSpecV927(
            "vision_mask_025", "vision", "mask", m, location=0.25
        ),
        PerturbationSpecV927(
            "vision_mask_065", "vision", "mask", m, location=0.65
        ),
    )


def _valid_time_positions(sample: torch.Tensor) -> torch.Tensor:
    """Find non-padding time positions in one [T,...] sample."""

    if sample.ndim == 0:
        return torch.ones(1, dtype=torch.bool, device=sample.device)
    if sample.ndim == 1:
        return sample.detach().abs() > 1e-12
    flat = sample.detach().abs().reshape(sample.shape[0], -1)
    return flat.amax(dim=1) > 1e-12


def _expand_time_mask(mask: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    shape = (mask.shape[0],) + (1,) * max(0, sample.ndim - 1)
    return mask.reshape(shape)


def _noise_for_sample(
    sample: torch.Tensor,
    valid: torch.Tensor,
    sample_index: int,
    spec: PerturbationSpecV927,
    config: PairedPerturbationConfigV927,
) -> torch.Tensor:
    output = sample.clone()
    if not bool(valid.any()):
        return output
    valid_values = sample[
        _expand_time_mask(valid, sample).expand_as(sample)
    ]
    scale = valid_values.float().std(unbiased=False)
    if not torch.isfinite(scale) or float(scale.item()) <= 1e-12:
        return output
    generator = torch.Generator(device="cpu")
    seed = (
        int(config.base_seed)
        + 1000003 * int(sample_index)
        + 9176 * int(spec.seed_offset)
    )
    generator.manual_seed(seed % (2**63 - 1))
    noise = torch.randn(
        tuple(sample.shape),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=sample.device, dtype=sample.dtype)
    mask = _expand_time_mask(valid, sample).expand_as(sample)
    output[mask] = output[mask] + (
        float(spec.magnitude) * scale.to(output) * noise[mask]
    )
    return output


def _mask_for_sample(
    sample: torch.Tensor,
    valid: torch.Tensor,
    spec: PerturbationSpecV927,
) -> torch.Tensor:
    output = sample.clone()
    valid_indices = torch.nonzero(valid, as_tuple=False).view(-1)
    count = int(valid_indices.numel())
    if count == 0:
        return output
    span = max(1, int(np.ceil(count * float(spec.magnitude))))
    span = min(span, count)
    center = int(round(float(spec.location) * max(0, count - 1)))
    start = min(max(0, center - span // 2), count - span)
    selected = valid_indices[start : start + span]
    output[selected] = 0
    return output


def _perturb_tensor(
    values: torch.Tensor,
    sample_indices: Sequence[int],
    spec: PerturbationSpecV927,
    config: PairedPerturbationConfigV927,
) -> torch.Tensor:
    if spec.kind == "identity":
        return values
    output = values.clone()
    if spec.kind == "gain":
        return output * float(spec.magnitude)
    for row, sample_index in enumerate(sample_indices):
        valid = _valid_time_positions(values[row])
        if spec.kind == "noise":
            output[row] = _noise_for_sample(
                values[row], valid, int(sample_index), spec, config
            )
        elif spec.kind == "mask":
            output[row] = _mask_for_sample(values[row], valid, spec)
        else:
            raise ValueError(f"unknown perturbation kind: {spec.kind}")
    return output


def apply_paired_perturbation(
    audio: torch.Tensor,
    vision: torch.Tensor,
    sample_indices: Sequence[int],
    spec: PerturbationSpecV927,
    config: PairedPerturbationConfigV927,
):
    """Apply one fixed perturbation while keeping text unchanged."""

    if spec.modality == "none":
        return audio, vision
    if spec.modality == "audio":
        return (
            _perturb_tensor(audio, sample_indices, spec, config),
            vision,
        )
    if spec.modality == "vision":
        return (
            audio,
            _perturb_tensor(vision, sample_indices, spec, config),
        )
    raise ValueError(f"unknown perturbation modality: {spec.modality}")


def _stack_checkpoint_paths(stack_dir: Path) -> Dict[str, Path]:
    stack_dir = Path(stack_dir)
    return {
        "cfcompat": stack_dir / "cfcompat_student_best_inner_valid.pth",
        "strong_negative": stack_dir / "strong_negative_expert_v919.pth",
        "boundary": stack_dir / "boundary_expert_v919.pth",
        "positive": stack_dir / "positive_expert_v919.pth",
        "strong_positive": stack_dir / "strong_positive_expert_v919.pth",
    }


def _checkpoint_fingerprints(paths: Mapping[str, Path]):
    missing = [str(path) for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing V9.19 checkpoints: " + ", ".join(missing)
        )
    return {name: sha256(Path(path)) for name, path in paths.items()}


def _load_cfcompat(args, checkpoint: Path):
    model = MissingModalityWrapper(
        DLF(args).to(args.device),
        int(args.feature_dims[1]),
        int(args.feature_dims[2]),
    ).to(args.device)
    model.load_state_dict(
        _load_state(Path(checkpoint), args.device), strict=True
    )
    model.eval()
    return model


def _load_role(args, checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    config = dict(payload["role_config"])
    model = RoleConditionedDLF(
        args,
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        residual_max=float(config["residual_max"]),
    ).to(args.device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, str(payload["role"])


def _load_tail(checkpoint: Path, feature_dim: int, device):
    payload = torch.load(checkpoint, map_location="cpu")
    config = dict(payload["tail_config"])
    model = CachedTailResidualHeadV92(
        feature_dim=int(feature_dim),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        residual_max=float(config["residual_max"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, str(payload["role"])


def _loader(dataset, indices, config: PairedPerturbationConfigV927):
    return build_subset_loader(
        dataset,
        [int(value) for value in indices],
        int(config.batch_size),
        int(config.num_workers),
        False,
        0,
    )


def _ordered_rows(rows: Mapping[int, object], indices: Sequence[int], name: str):
    expected = [int(value) for value in indices]
    if set(rows) != set(expected):
        missing = sorted(set(expected) - set(rows))
        extra = sorted(set(rows) - set(expected))
        raise RuntimeError(
            f"{name} row alignment failed; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    return [rows[index] for index in expected]


@torch.no_grad()
def _collect_cfcompat_variants(
    args,
    dataset,
    indices,
    checkpoint,
    specs,
    config,
):
    model = _load_cfcompat(args, checkpoint)
    all_anchor, all_feature = [], []
    try:
        loader = _loader(dataset, indices, config)
        for spec in specs:
            rows = {}
            for batch in loader:
                batch_indices = [
                    int(value)
                    for value in batch["index"].view(-1).cpu().tolist()
                ]
                text = batch["text"].to(args.device)
                audio = batch["audio"].to(args.device)
                vision = batch["vision"].to(args.device)
                audio, vision = apply_paired_perturbation(
                    audio, vision, batch_indices, spec, config
                )
                mask = mode_to_mask(
                    "LAV",
                    len(batch_indices),
                    args.device,
                    audio.dtype,
                )
                output = model(text, audio, vision, mask)
                feature = function_space_feature(output).detach().cpu()
                prediction = output["output_logit"].detach().cpu().view(-1)
                for offset, index in enumerate(batch_indices):
                    rows[index] = (
                        prediction[offset].clone(),
                        feature[offset].clone(),
                    )
            ordered = _ordered_rows(rows, indices, f"cfcompat/{spec.name}")
            all_anchor.append(torch.stack([row[0] for row in ordered]))
            all_feature.append(torch.stack([row[1] for row in ordered]))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.stack(all_anchor), torch.stack(all_feature)


@torch.no_grad()
def _collect_role_variants(
    args,
    dataset,
    indices,
    checkpoint,
    specs,
    config,
):
    model, role = _load_role(args, checkpoint)
    if role not in ROLE_INDEX:
        raise ValueError(f"unexpected role checkpoint: {role}")
    region_index = ROLE_INDEX[role]
    predictions, confidences, corrections = [], [], []
    try:
        loader = _loader(dataset, indices, config)
        for spec in specs:
            rows = {}
            for batch in loader:
                batch_indices = [
                    int(value)
                    for value in batch["index"].view(-1).cpu().tolist()
                ]
                text = batch["text"].to(args.device)
                audio = batch["audio"].to(args.device)
                vision = batch["vision"].to(args.device)
                audio, vision = apply_paired_perturbation(
                    audio, vision, batch_indices, spec, config
                )
                output = model(text, audio, vision)
                pred = output["prediction"].detach().cpu().view(-1)
                confidence = (
                    output["region_probs"][:, region_index]
                    .detach()
                    .cpu()
                    .view(-1)
                )
                correction = output["correction"].detach().cpu().view(-1)
                for offset, index in enumerate(batch_indices):
                    rows[index] = (
                        pred[offset].clone(),
                        confidence[offset].clone(),
                        correction[offset].clone(),
                    )
            ordered = _ordered_rows(rows, indices, f"{role}/{spec.name}")
            predictions.append(torch.stack([row[0] for row in ordered]))
            confidences.append(torch.stack([row[1] for row in ordered]))
            corrections.append(torch.stack([row[2] for row in ordered]))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "role": role,
        "prediction": torch.stack(predictions),
        "confidence": torch.stack(confidences),
        "correction": torch.stack(corrections),
    }


@torch.no_grad()
def _collect_tail_variants(
    checkpoint,
    features,
    anchors,
    batch_size,
    device,
):
    model, role = _load_tail(checkpoint, int(features.shape[-1]), device)
    buffers = {
        key: []
        for key in ("prediction", "confidence", "correction")
    }
    try:
        for variant in range(features.shape[0]):
            local = {
                key: []
                for key in ("prediction", "confidence", "correction")
            }
            for start in range(0, features.shape[1], int(batch_size)):
                output = model(
                    features[
                        variant, start : start + int(batch_size)
                    ].float().to(device),
                    anchors[
                        variant, start : start + int(batch_size)
                    ].float().to(device),
                )
                local["prediction"].append(
                    output["prediction"].detach().cpu().view(-1)
                )
                local["confidence"].append(
                    output["applicability_prob"].detach().cpu().view(-1)
                )
                local["correction"].append(
                    output["correction"].detach().cpu().view(-1)
                )
            for key in local:
                buffers[key].append(torch.cat(local[key]))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "role": role,
        **{key: torch.stack(value) for key, value in buffers.items()},
    }


def _validate_identity(
    payload,
    actions,
    confidences,
    corrections,
    tolerance,
):
    pool = normalize_pool(payload)
    saved_actions = pool.actions.detach().cpu().float()
    saved_confidence = (
        torch.as_tensor(payload["expert_confidences"])
        .detach()
        .cpu()
        .float()
        .squeeze(-1)
    )
    saved_correction = (
        torch.as_tensor(payload["expert_corrections"])
        .detach()
        .cpu()
        .float()
        .squeeze(-1)
    )
    checks = {
        "actions": float(
            torch.max(torch.abs(actions[0].float() - saved_actions)).item()
        ),
        "confidences": float(
            torch.max(
                torch.abs(confidences[0].float() - saved_confidence)
            ).item()
        ),
        "corrections": float(
            torch.max(
                torch.abs(corrections[0].float() - saved_correction)
            ).item()
        ),
    }
    if max(checks.values()) > float(tolerance):
        raise RuntimeError(
            "V9.27 checkpoint reconstruction does not match saved V9.19 "
            f"pool: {checks}"
        )
    return checks


def collect_stack_perturbations(
    args,
    dataset,
    stack_dir: Path,
    pool_payload: Mapping[str, object],
    config: PairedPerturbationConfigV927,
    cache_path: Path | None = None,
    resume: bool = True,
):
    """Rerun one frozen V9.19 stack on identity + paired perturbations."""

    config.validate()
    stack_dir = Path(stack_dir)
    cache_path = (
        Path(cache_path)
        if cache_path is not None
        else stack_dir / "paired_perturbation_pool_v927.pth"
    )
    indices = [int(value) for value in pool_payload["sample_indices"]]
    specs = perturbation_specs(config)
    paths = _stack_checkpoint_paths(stack_dir)
    fingerprints = _checkpoint_fingerprints(paths)
    reusable_key = {
        "version": AUDIT_VERSION,
        "config": asdict(config),
        "perturbation_names": [spec.name for spec in specs],
        "sample_indices": indices,
        "checkpoint_sha256": fingerprints,
    }
    if resume and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if all(cached.get(key) == value for key, value in reusable_key.items()):
            print(
                f"V9.27 reusing perturbation cache: {cache_path}",
                flush=True,
            )
            return cached

    print(
        f"V9.27 collecting CFCompat perturbations: {stack_dir}",
        flush=True,
    )
    anchors, features = _collect_cfcompat_variants(
        args,
        dataset,
        indices,
        paths["cfcompat"],
        specs,
        config,
    )
    role_outputs = {}
    for role in ("boundary", "positive"):
        print(
            f"V9.27 collecting {role} perturbations: {stack_dir}",
            flush=True,
        )
        local = _collect_role_variants(
            args,
            dataset,
            indices,
            paths[role],
            specs,
            config,
        )
        if local["role"] != role:
            raise RuntimeError(f"role checkpoint mismatch: {role}")
        role_outputs[role] = local

    tail_outputs = {}
    for role in ("strong_negative", "strong_positive"):
        print(
            f"V9.27 collecting {role} perturbations: {stack_dir}",
            flush=True,
        )
        local = _collect_tail_variants(
            paths[role],
            features,
            anchors,
            config.batch_size,
            args.device,
        )
        if local["role"] != role:
            raise RuntimeError(f"tail checkpoint mismatch: {role}")
        tail_outputs[role] = local

    variants = len(specs)
    n = len(indices)
    actions = torch.full((variants, n, len(ACTION_NAMES)), float("nan"))
    confidences = torch.full(
        (variants, n, len(SPECIALIST_NAMES)), float("nan")
    )
    corrections = torch.full_like(confidences, float("nan"))
    actions[:, :, 0] = anchors

    for expert_index, expert in enumerate(SPECIALIST_NAMES):
        source = (
            role_outputs[expert]
            if expert in role_outputs
            else tail_outputs[expert]
        )
        action_index = ACTION_NAMES.index(expert)
        actions[:, :, action_index] = source["prediction"]
        confidences[:, :, expert_index] = source["confidence"]
        corrections[:, :, expert_index] = source["correction"]

    if any(
        not torch.isfinite(value).all()
        for value in (actions, confidences, corrections, features)
    ):
        raise FloatingPointError("non-finite V9.27 perturbation output")

    identity_check = _validate_identity(
        pool_payload,
        actions,
        confidences,
        corrections,
        config.identity_tolerance,
    )
    sample_ids = [
        canonical_sample_id(value) for value in pool_payload["sample_ids"]
    ]
    result = {
        **reusable_key,
        "method": "frozen_v919_stack_model_input_perturbation",
        "sample_ids": sample_ids,
        "group_ids": [str(value) for value in pool_payload["group_ids"]],
        "actions": actions.float(),
        "expert_confidences": confidences.float(),
        "expert_corrections": corrections.float(),
        "function_space": features.float(),
        "identity_reconstruction_max_abs": identity_check,
        "provenance": {
            "base_experts_trained": False,
            "base_experts_modified": False,
            "v921_weights_modified": False,
            "raw_waveform_or_pixels_perturbed": False,
            "pre_extracted_model_input_features_perturbed": True,
            "text_perturbed": False,
            "same_perturbation_for_experts_and_baseline": True,
            "labels_used_to_generate_perturbations": False,
            "router_or_action_selection_executed": False,
            "prediction_replacement_or_mixing_executed": False,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    return result


def weighted_baseline(actions, weights):
    value = np.asarray(
        actions.detach().cpu().numpy()
        if torch.is_tensor(actions)
        else actions,
        dtype=np.float64,
    )
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.ndim != 3 or value.shape[-1] != len(weight):
        raise ValueError("actions/weights shape mismatch")
    if not np.isfinite(value).all() or not np.isfinite(weight).all():
        raise FloatingPointError("non-finite weighted baseline input")
    return np.einsum("vna,a->vn", value, weight)


def build_relative_stability_features(
    perturbation_payload,
    baseline_predictions,
    expert: str,
    config: PairedPerturbationConfigV927,
):
    """Build exactly eight pre-registered paired-stability features."""

    if expert not in EXPERT_NAMES:
        raise ValueError(f"unknown expert: {expert}")
    actions = np.asarray(
        torch.as_tensor(perturbation_payload["actions"]).cpu().numpy(),
        dtype=np.float64,
    )
    confidences = np.asarray(
        torch.as_tensor(
            perturbation_payload["expert_confidences"]
        ).cpu().numpy(),
        dtype=np.float64,
    )
    baseline = np.asarray(baseline_predictions, dtype=np.float64)
    if actions.ndim != 3 or baseline.shape != actions.shape[:2]:
        raise ValueError("perturbation action/baseline shape mismatch")
    if actions.shape[0] < 2:
        raise ValueError("at least one non-identity perturbation is required")

    action_index = EXPERT_ACTION_INDEX[expert]
    expert_index = EXPERT_NAMES.index(expert)
    original_expert = actions[0, :, action_index]
    perturbed_expert = actions[1:, :, action_index]
    original_baseline = baseline[0]
    perturbed_baseline = baseline[1:]
    original_confidence = confidences[0, :, expert_index]
    perturbed_confidence = confidences[1:, :, expert_index]

    expert_std = perturbed_expert.std(axis=0)
    baseline_std = perturbed_baseline.std(axis=0)
    log_std_ratio = np.log(
        (expert_std + float(config.ratio_epsilon))
        / (baseline_std + float(config.ratio_epsilon))
    )
    log_std_ratio = np.clip(log_std_ratio, -8.0, 8.0)

    original_gap = original_expert - original_baseline
    perturbed_gap = perturbed_expert - perturbed_baseline
    relative_gap_std = perturbed_gap.std(axis=0)

    epsilon = float(config.sign_epsilon)

    def sign(value):
        return np.where(
            value > epsilon,
            1,
            np.where(value < -epsilon, -1, 0),
        )

    original_sign = sign(original_gap)
    perturbed_sign = sign(perturbed_gap)
    relative_direction_flip_rate = (
        perturbed_sign != original_sign[None, :]
    ).mean(axis=0)

    original_applicable = original_confidence >= 0.5
    perturbed_applicable = perturbed_confidence >= 0.5
    applicability_flip_rate = (
        perturbed_applicable != original_applicable[None, :]
    ).mean(axis=0)

    expert_max_change = np.max(
        np.abs(perturbed_expert - original_expert[None, :]), axis=0
    )
    baseline_max_change = np.max(
        np.abs(perturbed_baseline - original_baseline[None, :]), axis=0
    )

    names = [
        "paired_expert_std",
        "paired_baseline_std",
        "paired_log_std_ratio",
        "paired_relative_gap_std",
        "paired_relative_direction_flip_rate",
        "paired_applicability_flip_rate",
        "paired_expert_max_change",
        "paired_baseline_max_change",
    ]
    matrix = np.column_stack(
        [
            expert_std,
            baseline_std,
            log_std_ratio,
            relative_gap_std,
            relative_direction_flip_rate,
            applicability_flip_rate,
            expert_max_change,
            baseline_max_change,
        ]
    )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("non-finite paired stability feature")
    return {
        "matrix": matrix,
        "feature_names": names,
        "expert_std": expert_std,
        "baseline_std": baseline_std,
        "log_std_ratio": log_std_ratio,
        "relative_gap_std": relative_gap_std,
        "relative_direction_flip_rate": relative_direction_flip_rate,
        "applicability_flip_rate": applicability_flip_rate,
        "expert_max_change": expert_max_change,
        "baseline_max_change": baseline_max_change,
    }


__all__ = [
    "AUDIT_VERSION",
    "PRIMARY_METHOD",
    "STABILITY_ONLY_METHOD",
    "BASE_METHOD",
    "PerturbationSpecV927",
    "PairedPerturbationConfigV927",
    "perturbation_specs",
    "apply_paired_perturbation",
    "collect_stack_perturbations",
    "weighted_baseline",
    "build_relative_stability_features",
]
