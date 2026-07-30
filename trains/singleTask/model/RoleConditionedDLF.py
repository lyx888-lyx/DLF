"""Role-conditioned multimodal DLF expert used by V9.

The wrapper keeps the original DLF predictor as a strong anchor and learns a
bounded residual, an independent five-region distribution, and a detached
sample-risk head.  The risk head never back-propagates into the shared expert
representation, which prevents the auxiliary task from damaging sentiment
prediction.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .FSC_DLF import _FusionFeatureCapture, load_dlf_checkpoint


class RoleConditionedDLF(nn.Module):
    """Strong DLF anchor plus role-specialized residual and auxiliary heads."""

    def __init__(
        self,
        args,
        hidden_dim: int = 192,
        dropout: float = 0.15,
        residual_max: float = 0.45,
    ):
        super().__init__()
        from . import DLF as DLFModel

        self.backbone = getattr(DLFModel, "DLF")(args)
        feature_dim = int(self.backbone.out_layer.in_features)
        self.residual_max = float(residual_max)

        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.residual_head = nn.Linear(hidden_dim, 1)
        self.region_head = nn.Linear(hidden_dim, 5)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 2),
            nn.Linear(hidden_dim + 2, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )

        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.constant_(self.risk_head[-1].bias, -1.0)

        self._stage = "heads"
        self.set_stage("heads")

    def load_backbone_checkpoint(self, path, map_location=None):
        return load_dlf_checkpoint(self.backbone, path, map_location)

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)

    def _tail_modules(self) -> Iterable[nn.Module]:
        return (
            self.backbone.projector_l,
            self.backbone.projector_a,
            self.backbone.projector_v,
            self.backbone.projector_c,
            self.backbone.proj1,
            self.backbone.proj2,
            self.backbone.out_layer,
        )

    def set_stage(self, stage: str) -> None:
        """Configure a conservative specialization schedule.

        heads: frozen DLF, train only new heads.
        tail: train new heads and the final DLF fusion/prediction tail.
        non_bert: additionally train all non-BERT DLF parameters.
        """

        if stage not in {"heads", "tail", "non_bert"}:
            raise ValueError(f"Unknown V9 training stage: {stage}")
        self._stage = stage
        self._set_requires_grad(self.backbone, False)
        self._set_requires_grad(self.adapter, True)
        self._set_requires_grad(self.residual_head, True)
        self._set_requires_grad(self.region_head, True)
        self._set_requires_grad(self.risk_head, True)

        if stage == "tail":
            for module in self._tail_modules():
                self._set_requires_grad(module, True)
        elif stage == "non_bert":
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad_(not name.startswith("text_model."))

        if hasattr(self.backbone, "text_model"):
            self._set_requires_grad(self.backbone.text_model, False)
            self.backbone.text_model.eval()

    def set_train_mode(self) -> None:
        self.train()
        self.backbone.eval()
        if hasattr(self.backbone, "text_model"):
            self.backbone.text_model.eval()

    def parameter_groups(
        self,
        head_lr: float,
        tail_lr: float,
        backbone_lr: float,
    ) -> List[Dict[str, object]]:
        head_parameters = []
        tail_parameters = []
        other_parameters = []
        tail_ids = {
            id(parameter)
            for module in self._tail_modules()
            for parameter in module.parameters()
        }
        head_modules = (
            self.adapter,
            self.residual_head,
            self.region_head,
            self.risk_head,
        )
        head_ids = {
            id(parameter)
            for module in head_modules
            for parameter in module.parameters()
        }
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in head_ids:
                head_parameters.append(parameter)
            elif id(parameter) in tail_ids:
                tail_parameters.append(parameter)
            else:
                other_parameters.append(parameter)

        groups: List[Dict[str, object]] = []
        if head_parameters:
            groups.append({"params": head_parameters, "lr": float(head_lr)})
        if tail_parameters:
            groups.append({"params": tail_parameters, "lr": float(tail_lr)})
        if other_parameters:
            groups.append({"params": other_parameters, "lr": float(backbone_lr)})
        if not groups:
            raise RuntimeError("V9 configured no trainable parameters.")
        return groups

    def forward(self, text, audio, vision) -> Dict[str, torch.Tensor]:
        capture = _FusionFeatureCapture(self.backbone)
        try:
            backbone_output = self.backbone(text, audio, vision)
            if capture.value is None:
                raise RuntimeError("V9 fusion feature hook did not capture a tensor.")
            feature = capture.value
        finally:
            capture.close()

        latent = self.adapter(feature)
        base_prediction = backbone_output["output_logit"]
        correction = self.residual_max * torch.tanh(self.residual_head(latent))
        prediction = base_prediction + correction

        region_logits = self.region_head(latent)
        region_probs = torch.softmax(region_logits, dim=1)
        centers = prediction.new_tensor((-2.25, -1.0, 0.0, 1.0, 2.25))
        region_expected = (region_probs * centers.view(1, -1)).sum(
            dim=1, keepdim=True
        )

        risk_features = torch.cat(
            [
                latent.detach(),
                prediction.detach(),
                region_probs.detach().amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        predicted_abs_error = F.softplus(self.risk_head(risk_features))

        return {
            "backbone": backbone_output,
            "feature": feature,
            "latent": latent,
            "base_prediction": base_prediction,
            "prediction": prediction,
            "correction": correction,
            "region_logits": region_logits,
            "region_probs": region_probs,
            "region_expected": region_expected,
            "predicted_abs_error": predicted_abs_error,
        }
