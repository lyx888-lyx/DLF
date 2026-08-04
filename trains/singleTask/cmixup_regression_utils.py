"Utilities for mode-consistent C-Mixup on CFCompatKD fusion features."

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cfcompat_stability_utils import preserve_rng_state
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_mode_consistent_cmixup_v1"
DEFAULT_ALPHA = 2.0
DEFAULT_BANDWIDTH = 0.5
DEFAULT_MIX_WEIGHT = 1.0
CMIXUP_SEED_OFFSET = 32452843


@dataclass(frozen=True)
class ModeConsistentMixBatch:
    partner_indices: torch.Tensor
    lambdas: torch.Tensor
    mixed_labels: torch.Tensor
    active_mask: torch.Tensor
    mean_partner_label_distance: float
    mean_lambda: float
    active_fraction: float
    mode_counts: Dict[str, int]
    partner_sha256: str


class FusionInputCapture:
    """Capture the differentiable input to DLF.proj1 without changing DLF."""

    def __init__(self, module: torch.nn.Module):
        self._queue: List[torch.Tensor] = []
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
            raise RuntimeError("DLF proj1 pre-hook expected one tensor input.")
        self._queue.append(inputs[0])

    def pop(self) -> torch.Tensor:
        if not self._queue:
            raise RuntimeError("No fusion feature was captured.")
        return self._queue.pop(0)

    def assert_empty(self) -> None:
        if self._queue:
            raise RuntimeError(
                "Unconsumed fusion captures remain: {}".format(len(self._queue))
            )

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class ModeConsistentCMixupSampler:
    """Batch-local label-KDE partner sampling restricted to one missing mode."""

    def __init__(
        self,
        seed: int,
        alpha: float = DEFAULT_ALPHA,
        bandwidth: float = DEFAULT_BANDWIDTH,
    ):
        if float(alpha) <= 0:
            raise ValueError("C-Mixup alpha must be positive.")
        if float(bandwidth) <= 0:
            raise ValueError("C-Mixup bandwidth must be positive.")
        self.alpha = float(alpha)
        self.bandwidth = float(bandwidth)
        self.rng = np.random.default_rng(int(seed) + CMIXUP_SEED_OFFSET)
        self._partner_hash = hashlib.sha256()
        self.draw_count = 0

    @staticmethod
    def _probabilities(labels: np.ndarray, bandwidth: float) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        if labels.size < 2:
            raise ValueError("At least two labels are required for C-Mixup.")
        distance_squared = (labels[:, None] - labels[None, :]) ** 2
        logits = -distance_squared / (2.0 * float(bandwidth) ** 2)
        np.fill_diagonal(logits, -np.inf)
        row_max = np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits - row_max)
        np.fill_diagonal(weights, 0.0)
        denominators = weights.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(denominators)) or np.any(denominators <= 0):
            raise RuntimeError("C-Mixup label-KDE probabilities are invalid.")
        return weights / denominators

    def sample(
        self,
        labels: torch.Tensor,
        modes: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> ModeConsistentMixBatch:
        labels_flat = labels.detach().view(-1).cpu().numpy().astype(np.float64)
        modes = [str(mode) for mode in modes]
        if len(modes) != labels_flat.size:
            raise ValueError("Mode count must equal label count.")
        unknown = sorted(set(modes) - set(MISSING_MODES))
        if unknown:
            raise ValueError("Unsupported missing modes: {}".format(unknown))

        batch_size = labels_flat.size
        partner = np.arange(batch_size, dtype=np.int64)
        lambdas = np.ones(batch_size, dtype=np.float32)
        active = np.zeros(batch_size, dtype=bool)
        mode_counts = {mode: int(sum(item == mode for item in modes)) for mode in MISSING_MODES}

        mode_array = np.asarray(modes, dtype=object)
        for mode in MISSING_MODES:
            group = np.flatnonzero(mode_array == mode)
            if group.size < 2:
                continue
            local_probabilities = self._probabilities(
                labels_flat[group], self.bandwidth
            )
            for local_row, global_index in enumerate(group):
                local_partner = int(
                    self.rng.choice(group.size, p=local_probabilities[local_row])
                )
                global_partner = int(group[local_partner])
                if global_partner == int(global_index):
                    raise RuntimeError("C-Mixup sampled a self partner.")
                partner[global_index] = global_partner
                lambdas[global_index] = float(
                    self.rng.beta(self.alpha, self.alpha)
                )
                active[global_index] = True

        for index in np.flatnonzero(active):
            if modes[index] != modes[int(partner[index])]:
                raise RuntimeError("C-Mixup partner crossed missing modes.")

        partner_bytes = partner.astype("<i8", copy=False).tobytes()
        lambda_bytes = lambdas.astype("<f4", copy=False).tobytes()
        active_bytes = active.astype(np.uint8, copy=False).tobytes()
        self._partner_hash.update(partner_bytes + lambda_bytes + active_bytes)
        self.draw_count += int(active.sum())

        partner_tensor = torch.as_tensor(partner, device=device, dtype=torch.long)
        lambda_tensor = torch.as_tensor(
            lambdas, device=device, dtype=dtype
        ).view(-1, 1)
        active_tensor = torch.as_tensor(active, device=device, dtype=torch.bool)
        mixed_labels = (
            lambda_tensor * labels
            + (1.0 - lambda_tensor) * labels.index_select(0, partner_tensor)
        )

        if active.any():
            distances = np.abs(labels_flat - labels_flat[partner])
            mean_distance = float(distances[active].mean())
            mean_lambda = float(lambdas[active].mean())
        else:
            mean_distance = 0.0
            mean_lambda = 1.0

        return ModeConsistentMixBatch(
            partner_indices=partner_tensor,
            lambdas=lambda_tensor,
            mixed_labels=mixed_labels,
            active_mask=active_tensor,
            mean_partner_label_distance=mean_distance,
            mean_lambda=mean_lambda,
            active_fraction=float(active.mean()) if batch_size else 0.0,
            mode_counts=mode_counts,
            partner_sha256=self._partner_hash.hexdigest(),
        )

    def hexdigest(self) -> str:
        return self._partner_hash.hexdigest()


def mix_fusion_features(
    fusion: torch.Tensor,
    partner_indices: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    if fusion.ndim != 2:
        raise ValueError("Fusion features must have shape [batch, feature].")
    if partner_indices.ndim != 1 or partner_indices.size(0) != fusion.size(0):
        raise ValueError("Partner indices must have shape [batch].")
    if lambdas.shape != (fusion.size(0), 1):
        raise ValueError("Lambdas must have shape [batch, 1].")
    return (
        lambdas * fusion
        + (1.0 - lambdas) * fusion.index_select(0, partner_indices)
    )


def predict_from_dlf_fusion(
    backbone: torch.nn.Module,
    fusion: torch.Tensor,
) -> torch.Tensor:
    """Run the unchanged DLF final residual MLP from a supplied fusion tensor."""
    hidden = backbone.proj1(fusion)
    hidden = F.relu(hidden, inplace=True)
    hidden = F.dropout(
        hidden,
        p=backbone.output_dropout,
        training=backbone.training,
    )
    hidden = backbone.proj2(hidden)
    hidden = hidden + fusion
    return backbone.out_layer(hidden)


def compute_mode_consistent_cmixup_loss(
    student: torch.nn.Module,
    full_fusion: torch.Tensor,
    missing_fusion: torch.Tensor,
    labels: torch.Tensor,
    modes: Sequence[str],
    sampler: ModeConsistentCMixupSampler,
    criterion: torch.nn.Module,
):
    if full_fusion.shape != missing_fusion.shape:
        raise ValueError("Full and missing fusion features must have equal shape.")
    if full_fusion.size(0) != labels.size(0):
        raise ValueError("Fusion batch size does not match labels.")

    batch = sampler.sample(
        labels=labels,
        modes=modes,
        device=full_fusion.device,
        dtype=full_fusion.dtype,
    )
    if not bool(batch.active_mask.any()):
        zero = (full_fusion.sum() + missing_fusion.sum()) * 0.0
        diagnostics = {
            "mix_active_fraction": 0.0,
            "mix_mean_lambda": 1.0,
            "mix_mean_partner_label_distance": 0.0,
            "mix_full_loss": 0.0,
            "mix_missing_loss": 0.0,
            "mix_partner_sha256": batch.partner_sha256,
            **{
                "mix_{}_count".format(mode): int(batch.mode_counts[mode])
                for mode in MISSING_MODES
            },
        }
        return zero, diagnostics, batch

    mixed_full = mix_fusion_features(
        full_fusion, batch.partner_indices, batch.lambdas
    )
    mixed_missing = mix_fusion_features(
        missing_fusion, batch.partner_indices, batch.lambdas
    )
    combined = torch.cat([mixed_full, mixed_missing], dim=0)

    # The extra regularizer receives a real dropout mask but does not shift the
    # frozen Stage-3 random trajectory used by later original forwards.
    with preserve_rng_state():
        combined_prediction = predict_from_dlf_fusion(
            student.backbone, combined
        )

    split = mixed_full.size(0)
    full_prediction = combined_prediction[:split]
    missing_prediction = combined_prediction[split:]
    mask = batch.active_mask
    full_loss = criterion(
        full_prediction[mask], batch.mixed_labels[mask]
    )
    missing_loss = criterion(
        missing_prediction[mask], batch.mixed_labels[mask]
    )
    loss = full_loss + missing_loss
    diagnostics = {
        "mix_active_fraction": batch.active_fraction,
        "mix_mean_lambda": batch.mean_lambda,
        "mix_mean_partner_label_distance": batch.mean_partner_label_distance,
        "mix_full_loss": float(full_loss.detach()),
        "mix_missing_loss": float(missing_loss.detach()),
        "mix_partner_sha256": batch.partner_sha256,
        **{
            "mix_{}_count".format(mode): int(batch.mode_counts[mode])
            for mode in MISSING_MODES
        },
    }
    return loss, diagnostics, batch


def aggregate_mix_diagnostics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        raise ValueError("No C-Mixup diagnostics were supplied.")
    numeric_keys = (
        "mix_active_fraction",
        "mix_mean_lambda",
        "mix_mean_partner_label_distance",
        "mix_full_loss",
        "mix_missing_loss",
    )
    result = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in numeric_keys
    }
    result["mix_partner_sha256"] = str(rows[-1]["mix_partner_sha256"])
    for mode in MISSING_MODES:
        key = "mix_{}_count".format(mode)
        result[key] = int(sum(int(row[key]) for row in rows))
    return result


def promotion_gate(candidate: dict, baseline: dict, epoch_rows) -> dict:
    """Frozen seed-level gate; positive gains mean the candidate is better."""
    baseline_missing = float(
        np.mean(
            [
                float(baseline["valid_{}_MAE".format(mode)])
                for mode in MISSING_MODES
            ]
        )
    )
    candidate_missing = float(
        np.mean(
            [
                float(candidate["valid_{}_MAE".format(mode)])
                for mode in MISSING_MODES
            ]
        )
    )
    gain_j = float(baseline["J_valid"]) - float(candidate["J_valid"])
    gain_lav = (
        float(baseline["valid_LAV_MAE"])
        - float(candidate["valid_LAV_MAE"])
    )
    missing_degradation = candidate_missing - baseline_missing
    supporting_epochs = int(
        sum(
            float(row["J_valid"])
            <= float(baseline["J_valid"]) - 0.003
            for row in epoch_rows
        )
    )
    checks = {
        "valid_J_gain_ge_0p005": gain_j >= 0.005,
        "valid_LAV_MAE_gain_ge_0p005": gain_lav >= 0.005,
        "missing_macro_degradation_le_0p002": missing_degradation <= 0.002,
        "at_least_two_supporting_epochs": supporting_epochs >= 2,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "gain_valid_J": gain_j,
        "gain_valid_LAV_MAE": gain_lav,
        "missing_macro_degradation": missing_degradation,
        "supporting_epoch_count": supporting_epochs,
        "required_gain_valid_J": 0.005,
        "required_gain_valid_LAV_MAE": 0.005,
        "max_missing_macro_degradation": 0.002,
        "required_supporting_epochs": 2,
    }
