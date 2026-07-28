"""Safe independent text/audio/vision experts for V8.

The expert consumes exactly one modality and returns sequence and pooled
representations, a scalar sentiment prediction, and a bounded sample-wise error
score.  The error head can be detached from the shared representation so the
auxiliary task cannot damage the sentiment encoder.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from ...subNets import BertTextEncoder


_MODALITY_TO_KEY = {
    "text": "text",
    "audio": "audio",
    "vision": "vision",
}


class UnimodalExpertV8(nn.Module):
    """Modality-specific expert with a direction-consistent error head."""

    def __init__(
        self,
        args,
        modality: str,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_multiplier: int = 4,
        dropout: float = 0.25,
        max_length: int = 512,
        layer_fusion: str = "final",
        pooling: str = "auto",
        finetune_text_encoder: bool = True,
    ):
        super().__init__()
        modality = str(modality).lower()
        if modality not in _MODALITY_TO_KEY:
            raise ValueError(f"Unsupported modality: {modality}")
        if layer_fusion not in ("final", "mid_last"):
            raise ValueError("layer_fusion must be 'final' or 'mid_last'.")
        if pooling not in ("auto", "cls", "mean"):
            raise ValueError("pooling must be one of: auto, cls, mean.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        if layer_fusion == "mid_last" and num_layers < 2:
            raise ValueError("mid_last fusion requires at least two layers.")

        self.modality = modality
        self.batch_key = _MODALITY_TO_KEY[modality]
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.ffn_multiplier = int(ffn_multiplier)
        self.dropout = float(dropout)
        self.max_length = int(max_length)
        self.layer_fusion = layer_fusion
        self.finetune_text_encoder = bool(finetune_text_encoder)
        self.use_bert = bool(modality == "text" and args.use_bert)
        self.pooling = (
            "cls" if pooling == "auto" and self.use_bert
            else "mean" if pooling == "auto"
            else pooling
        )
        if self.pooling == "cls" and not self.use_bert:
            raise ValueError("CLS pooling is only valid for BERT text inputs.")

        feature_dims = tuple(int(value) for value in args.feature_dims)
        self.input_dim = {
            "text": feature_dims[0],
            "audio": feature_dims[1],
            "vision": feature_dims[2],
        }[modality]

        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=self.finetune_text_encoder,
                transformers=args.transformers,
                pretrained=args.pretrained,
            )

        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_length, self.hidden_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * self.ffn_multiplier,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.hidden_dim)

        if self.layer_fusion == "mid_last":
            self.mid_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.last_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.layer_fusion_projection = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )

        head_hidden = max(32, self.hidden_dim // 2)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(head_hidden, 1),
        )
        self.error_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(head_hidden, 1),
        )

        self.register_buffer("normalizer_mean", torch.zeros(self.input_dim))
        self.register_buffer("normalizer_std", torch.ones(self.input_dim))
        self.register_buffer("normalizer_enabled", torch.tensor(False))
        self.register_buffer("error_scale", torch.tensor(1.0))

    def model_profile(self) -> Dict[str, object]:
        return {
            "modality": self.modality,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_multiplier": self.ffn_multiplier,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "layer_fusion": self.layer_fusion,
            "pooling": self.pooling,
            "use_bert": self.use_bert,
            "finetune_text_encoder": self.finetune_text_encoder,
        }

    def set_normalizer(
        self,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> None:
        """Install feature statistics computed exclusively from Train."""
        if mean is None or std is None:
            self.normalizer_enabled.fill_(False)
            return
        mean = mean.detach().float().view(-1)
        std = std.detach().float().view(-1).clamp_min(1e-5)
        if mean.numel() != self.input_dim or std.numel() != self.input_dim:
            raise ValueError(
                f"Normalizer dimension mismatch: expected {self.input_dim}, "
                f"got mean={mean.numel()} std={std.numel()}"
            )
        self.normalizer_mean.copy_(mean)
        self.normalizer_std.copy_(std)
        self.normalizer_enabled.fill_(True)

    def set_error_scale(self, value: float) -> None:
        self.error_scale.fill_(max(float(value), 1e-4))

    def _prepare_input(
        self,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sequence, valid-token mask and all-missing sample flags."""
        if self.use_bert:
            if value.ndim != 3 or value.shape[1] < 2:
                raise ValueError(
                    "BERT text input must have shape [batch, 3, sequence]."
                )
            mask = value[:, 1, :].bool()
            sequence = self.text_model(value)
        else:
            if value.ndim != 3:
                raise ValueError(
                    f"{self.modality} input must have shape [batch, time, dim]."
                )
            sequence = value.float()
            finite = torch.isfinite(sequence).all(dim=-1)
            nonzero = sequence.abs().sum(dim=-1) > 0
            mask = finite & nonzero

        sequence = torch.nan_to_num(sequence.float())
        all_missing = ~mask.any(dim=1)
        effective_mask = mask.clone()
        if all_missing.any():
            effective_mask[all_missing, 0] = True

        if bool(self.normalizer_enabled.item()):
            sequence = (
                sequence - self.normalizer_mean.view(1, 1, -1)
            ) / self.normalizer_std.view(1, 1, -1)

        sequence = sequence.masked_fill(~effective_mask.unsqueeze(-1), 0.0)
        if all_missing.any():
            sequence[all_missing] = 0.0
        return sequence, effective_mask, all_missing

    @staticmethod
    def _masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(sequence.dtype)
        return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode(self, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        sequence, mask, all_missing = self._prepare_input(value)
        length = sequence.shape[1]
        if length > self.position_embedding.shape[1]:
            raise ValueError(
                f"Sequence length {length} exceeds max_length "
                f"{self.position_embedding.shape[1]}."
            )

        hidden = self.input_projection(sequence)
        hidden = hidden + self.position_embedding[:, :length]
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)

        layer_outputs = []
        padding_mask = ~mask
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
            layer_outputs.append(hidden)

        last = self.final_norm(hidden)
        last = last.masked_fill(~mask.unsqueeze(-1), 0.0)
        if self.layer_fusion == "mid_last":
            middle_index = max(0, (len(layer_outputs) - 1) // 2)
            middle = layer_outputs[middle_index]
            fused = torch.cat([
                self.mid_projection(middle),
                self.last_projection(last),
            ], dim=-1)
            sequence_feature = self.layer_fusion_projection(fused)
            sequence_feature = sequence_feature.masked_fill(
                ~mask.unsqueeze(-1), 0.0
            )
        else:
            sequence_feature = last

        if self.pooling == "cls":
            pooled = sequence_feature[:, 0]
        else:
            pooled = self._masked_mean(sequence_feature, mask)
        pooled = pooled.masked_fill(all_missing.unsqueeze(-1), 0.0)

        return {
            "sequence": sequence_feature,
            "pooled": pooled,
            "mask": mask,
            "all_missing": all_missing,
        }

    def forward(
        self,
        value: torch.Tensor,
        detach_uncertainty_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode(value)
        pooled = encoded["pooled"]
        prediction = self.prediction_head(pooled)
        uncertainty_feature = (
            pooled.detach() if detach_uncertainty_features else pooled
        )
        uncertainty_logit = self.error_head(uncertainty_feature)
        uncertainty = torch.sigmoid(uncertainty_logit)
        return {
            **encoded,
            "prediction": prediction,
            "uncertainty_logit": uncertainty_logit,
            "uncertainty": uncertainty,
        }

    def error_target(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """High target values always mean larger absolute prediction error."""
        return torch.tanh(
            torch.abs(labels - prediction.detach())
            / self.error_scale.clamp_min(1e-4)
        )

    def prediction_parameters(self) -> Iterable[nn.Parameter]:
        excluded = {id(parameter) for parameter in self.error_head.parameters()}
        return (
            parameter for parameter in self.parameters()
            if id(parameter) not in excluded
        )

    def configure_stage(self, stage: str) -> None:
        stage = str(stage).lower()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        if stage == "prediction":
            for parameter in self.prediction_parameters():
                parameter.requires_grad_(True)
        elif stage == "uncertainty":
            for parameter in self.error_head.parameters():
                parameter.requires_grad_(True)
        elif stage in ("joint_detached", "joint_shared"):
            for parameter in self.parameters():
                parameter.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training stage: {stage}")

        if self.use_bert and not self.finetune_text_encoder:
            for parameter in self.text_model.parameters():
                parameter.requires_grad_(False)

    def set_stage_mode(self, stage: str) -> None:
        stage = str(stage).lower()
        if stage == "uncertainty":
            self.eval()
            self.error_head.train()
        else:
            self.train()
            if self.use_bert and not self.finetune_text_encoder:
                self.text_model.eval()
