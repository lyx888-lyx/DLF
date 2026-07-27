"""Recoverability-aware posterior DLF with active modality acquisition.

The model wraps a DLF backbone and evaluates four observable modality states:
LAV, LA, LV and L.  A shared recoverability encoder projects each observable
state into the latent space of the complete view.  A conditional innovation
posterior then represents multiple plausible complete-state continuations.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import DLF as DLFModel


MODE_NAMES: Tuple[str, ...] = ("lav", "la", "lv", "l")
MODE_TO_ID = {name: index for index, name in enumerate(MODE_NAMES)}


class _FusionFeatureCapture:
    """Capture the tensor directly before the DLF scalar output layer."""

    def __init__(self, backbone: nn.Module):
        self.value = None
        self.handle = backbone.out_layer.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        self.value = inputs[0]

    def close(self):
        self.handle.remove()


def _expert_context(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    matrix = torch.cat(
        [
            output["logits_l_hetero"],
            output["logits_a_hetero"],
            output["logits_v_hetero"],
            output["logits_c"],
            output["output_logit"],
        ],
        dim=1,
    )
    fusion = matrix[:, -1:]
    differences = matrix[:, :-1] - fusion
    mean = matrix.mean(dim=1, keepdim=True)
    std = matrix.std(dim=1, keepdim=True, unbiased=False)
    minimum = matrix.min(dim=1, keepdim=True).values
    maximum = matrix.max(dim=1, keepdim=True).values
    return torch.cat(
        [matrix, differences, mean, std, minimum, maximum, maximum - minimum],
        dim=1,
    )


def weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return a weighted median for each row of ``values``."""

    order = values.argsort(dim=1)
    sorted_values = values.gather(1, order)
    sorted_weights = weights.gather(1, order)
    indices = (sorted_weights.cumsum(dim=1) < 0.5).sum(dim=1)
    indices = indices.clamp(max=values.size(1) - 1)
    return sorted_values.gather(1, indices.view(-1, 1))


class RecoverableInnovationPosterior(nn.Module):
    """Shared recoverability projector and conditional innovation posterior."""

    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        mode_count: int = 4,
        latent_dim: int = 128,
        posterior_components: int = 5,
        dropout: float = 0.20,
        residual_max: float = 1.5,
        latent_scale: float = 1.0,
    ):
        super().__init__()
        if posterior_components < 3:
            raise ValueError("posterior_components must be at least 3")
        self.mode_count = int(mode_count)
        self.latent_dim = int(latent_dim)
        self.posterior_components = int(posterior_components)
        self.residual_max = float(residual_max)
        self.latent_scale = float(latent_scale)

        self.mode_embedding = nn.Embedding(self.mode_count, 16)
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(latent_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(latent_dim), int(latent_dim)),
        )
        self.target_projection = nn.Linear(
            int(feature_dim), int(latent_dim), bias=False
        )
        nn.init.orthogonal_(self.target_projection.weight)
        for parameter in self.target_projection.parameters():
            parameter.requires_grad_(False)
        self.target_norm = nn.LayerNorm(int(latent_dim), elementwise_affine=False)
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(int(context_dim)),
            nn.Linear(int(context_dim), 64),
            nn.GELU(),
            nn.Dropout(float(dropout * 0.5)),
            nn.Linear(64, 48),
            nn.GELU(),
        )
        self.recoverable = nn.Sequential(
            nn.Linear(int(latent_dim) + 48 + 16, int(latent_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(latent_dim), int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
        )
        self.innovation_head = nn.Sequential(
            nn.Linear(int(latent_dim), int(latent_dim)),
            nn.GELU(),
            nn.Linear(
                int(latent_dim),
                self.posterior_components * int(latent_dim),
            ),
        )
        self.weight_head = nn.Sequential(
            nn.Linear(int(latent_dim) + 48, int(latent_dim)),
            nn.GELU(),
            nn.Linear(int(latent_dim), self.posterior_components),
        )
        self.residual_decoder = nn.Sequential(
            nn.Linear(int(latent_dim), int(latent_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout * 0.5)),
            nn.Linear(int(latent_dim), 1),
        )
        self.sign_head = nn.Linear(int(latent_dim), 1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(int(latent_dim) + 48, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.voi_head = nn.Sequential(
            nn.Linear(int(latent_dim) + 48 + 16, 96),
            nn.GELU(),
            nn.Dropout(float(dropout * 0.5)),
            nn.Linear(96, 3),
        )

        initial = torch.linspace(-0.45, 0.45, self.posterior_components)
        self.component_bias = nn.Parameter(initial)
        nn.init.zeros_(self.residual_decoder[-1].weight)
        nn.init.zeros_(self.residual_decoder[-1].bias)

    def forward(
        self,
        fusion_feature: torch.Tensor,
        context: torch.Tensor,
        anchor: torch.Tensor,
        mode_ids: torch.Tensor,
        full_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feature = self.feature_encoder(fusion_feature)
        context_latent = self.context_encoder(context)
        mode_latent = self.mode_embedding(mode_ids.view(-1))
        recoverable = self.recoverable(
            torch.cat([feature, context_latent, mode_latent], dim=1)
        )
        target_full = self.target_norm(self.target_projection(full_feature))
        innovations = self.innovation_head(recoverable).view(
            -1, self.posterior_components, self.latent_dim
        )
        innovations = self.latent_scale * torch.tanh(innovations)
        latent_candidates = recoverable.unsqueeze(1) + innovations
        residual = self.residual_decoder(latent_candidates).squeeze(-1)
        residual = self.residual_max * torch.tanh(
            residual + self.component_bias.view(1, -1)
        )
        candidate_predictions = anchor + residual
        logits = self.weight_head(torch.cat([recoverable, context_latent], dim=1))
        weights = F.softmax(logits, dim=1)
        mean_prediction = (weights * candidate_predictions).sum(dim=1, keepdim=True)
        median_prediction = weighted_median(candidate_predictions, weights)
        expected_latent = (weights.unsqueeze(-1) * latent_candidates).sum(dim=1)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        spread = (
            weights
            * (candidate_predictions - mean_prediction).square()
        ).sum(dim=1, keepdim=True).sqrt()
        predicted_scale = F.softplus(
            self.uncertainty_head(torch.cat([recoverable, context_latent], dim=1))
        )
        voi = self.voi_head(
            torch.cat([recoverable, context_latent, mode_latent], dim=1)
        )
        return {
            "recoverable": recoverable,
            "target_full": target_full,
            "innovations": innovations,
            "latent_candidates": latent_candidates,
            "expected_latent": expected_latent,
            "candidate_predictions": candidate_predictions,
            "component_residuals": residual,
            "component_logits": logits,
            "component_weights": weights,
            "mean_prediction": mean_prediction,
            "median_prediction": median_prediction,
            "entropy": entropy,
            "spread": spread,
            "predicted_scale": predicted_scale,
            "sign_logit": self.sign_head(recoverable),
            "voi": voi,
        }


class RADIANTDLF(nn.Module):
    """DLF backbone plus recoverability-aware counterfactual posterior."""

    def __init__(
        self,
        args,
        latent_dim: int = 128,
        posterior_components: int = 5,
        dropout: float = 0.20,
        residual_max: float = 1.5,
        latent_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        context_dim = 14
        self.posterior = RecoverableInnovationPosterior(
            feature_dim=feature_dim,
            context_dim=context_dim,
            mode_count=len(MODE_NAMES),
            latent_dim=latent_dim,
            posterior_components=posterior_components,
            dropout=dropout,
            residual_max=residual_max,
            latent_scale=latent_scale,
        )
        self._backbone_frozen = False

    def load_backbone_checkpoint(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return self.backbone.load_state_dict(state, strict=False)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self._backbone_frozen = True

    def unfreeze_backbone_tail(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        modules: Iterable[nn.Module] = (
            self.backbone.projector_l,
            self.backbone.projector_a,
            self.backbone.projector_v,
            self.backbone.projector_c,
            self.backbone.proj1,
            self.backbone.proj2,
            self.backbone.out_layer,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self._backbone_frozen = False

    def set_train_mode(self):
        self.train()
        self.backbone.eval()
        if hasattr(self.backbone, "text_model"):
            self.backbone.text_model.eval()

    @staticmethod
    def _mask_inputs(
        audio: torch.Tensor,
        vision: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "lav":
            return audio, vision
        if mode == "la":
            return audio, torch.zeros_like(vision)
        if mode == "lv":
            return torch.zeros_like(audio), vision
        if mode == "l":
            return torch.zeros_like(audio), torch.zeros_like(vision)
        raise ValueError(f"Unknown modality mode: {mode}")

    def _backbone_view(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        masked_audio, masked_vision = self._mask_inputs(audio, vision, mode)
        capture = _FusionFeatureCapture(self.backbone)
        try:
            output = self.backbone(text, masked_audio, masked_vision)
            if capture.value is None:
                raise RuntimeError("DLF fusion feature hook did not capture a tensor")
            feature = capture.value
        finally:
            capture.close()
        return {
            "backbone": output,
            "fusion_feature": feature,
            "context": _expert_context(output),
            "anchor": output["output_logit"],
        }

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        modes: Tuple[str, ...] = MODE_NAMES,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        views: "OrderedDict[str, Dict[str, torch.Tensor]]" = OrderedDict()
        original_use_bert = bool(self.backbone.use_bert)
        encoded_text = text
        if original_use_bert:
            encoded_text = self.backbone.text_model(text)
            self.backbone.use_bert = False
        try:
            full = self._backbone_view(encoded_text, audio, vision, "lav")
            views["lav"] = full
            for mode in modes:
                if mode == "lav":
                    continue
                views[mode] = self._backbone_view(encoded_text, audio, vision, mode)
        finally:
            self.backbone.use_bert = original_use_bert
        full_feature = full["fusion_feature"]
        batch_size = text.size(0)
        for mode, view in views.items():
            mode_ids = torch.full(
                (batch_size,),
                MODE_TO_ID[mode],
                dtype=torch.long,
                device=text.device,
            )
            posterior = self.posterior(
                fusion_feature=view["fusion_feature"],
                context=view["context"],
                anchor=view["anchor"],
                mode_ids=mode_ids,
                full_feature=full_feature,
            )
            view.update(posterior)
            view["mode_ids"] = mode_ids
        return views
