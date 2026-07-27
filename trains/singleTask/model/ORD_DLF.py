import torch
import torch.nn as nn
import torch.nn.functional as F

from .DLF import DLF as BaseDLF


class CandidateEnergyHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=192, candidate_dim=64, dropout=0.20):
        super().__init__()
        self.context = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.candidate = nn.Sequential(
            nn.Linear(5, candidate_dim),
            nn.GELU(),
            nn.Linear(candidate_dim, candidate_dim),
            nn.GELU(),
        )
        self.energy = nn.Sequential(
            nn.Linear(hidden_dim + candidate_dim + candidate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.context_to_candidate = nn.Linear(hidden_dim, candidate_dim)

    def forward(self, fusion_feature, fusion_prediction, offsets):
        batch_size = fusion_prediction.size(0)
        candidate_count = offsets.numel()
        context = self.context(fusion_feature)
        context_candidate = self.context_to_candidate(context)

        offset = offsets.view(1, candidate_count).expand(batch_size, -1)
        fusion = fusion_prediction.view(batch_size, 1).expand(-1, candidate_count)
        candidate_value = fusion + offset
        descriptor = torch.stack(
            [candidate_value, offset, offset.abs(), offset.sign(), fusion], dim=-1
        )
        candidate = self.candidate(descriptor)
        context_expanded = context.unsqueeze(1).expand(-1, candidate_count, -1)
        interaction = candidate * context_candidate.unsqueeze(1)
        logits = self.energy(
            torch.cat([context_expanded, candidate, interaction], dim=-1)
        ).squeeze(-1)
        return logits, candidate_value


class OracleRegretDLF(nn.Module):
    """Joint DLF backbone and candidate-conditioned oracle-regret student."""

    def __init__(
        self,
        args,
        offsets=(-0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 0.75),
        router_hidden_dim=192,
        candidate_dim=64,
        router_dropout=0.20,
    ):
        super().__init__()
        self.backbone = BaseDLF(args)
        self.register_buffer('candidate_offsets', torch.tensor(offsets, dtype=torch.float32))
        self.energy_head = CandidateEnergyHead(
            self.backbone.out_layer.in_features,
            hidden_dim=router_hidden_dim,
            candidate_dim=candidate_dim,
            dropout=router_dropout,
        )
        self._fusion_feature = None
        self._feature_hook = self.backbone.out_layer.register_forward_pre_hook(
            self._capture_fusion_feature
        )

    def _capture_fusion_feature(self, module, inputs):
        del module
        self._fusion_feature = inputs[0]

    @property
    def zero_candidate_index(self):
        return int(torch.argmin(self.candidate_offsets.abs()).item())

    def load_backbone_checkpoint(self, path, map_location='cpu'):
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and 'state_dict' in payload:
            payload = payload['state_dict']
        clean = {}
        for key, value in payload.items():
            if key.startswith('backbone.'):
                key = key[len('backbone.'):]
            clean[key] = value
        return self.backbone.load_state_dict(clean, strict=False)

    def candidate_probabilities(self, logits, temperature=1.0):
        return F.softmax(logits / max(float(temperature), 1e-4), dim=1)

    def route(self, logits, candidates, fusion_prediction, temperature=1.0, blend=1.0):
        probabilities = self.candidate_probabilities(logits, temperature)
        soft_prediction = (probabilities * candidates).sum(dim=1, keepdim=True)
        routed = fusion_prediction + float(blend) * (soft_prediction - fusion_prediction)
        return probabilities, routed

    def forward(self, text, audio, video, temperature=1.0, blend=1.0):
        self._fusion_feature = None
        output = self.backbone(text, audio, video)
        if self._fusion_feature is None:
            raise RuntimeError('Failed to capture the DLF fusion feature.')
        fusion_prediction = output['output_logit']
        logits, candidates = self.energy_head(
            self._fusion_feature, fusion_prediction, self.candidate_offsets
        )
        probabilities, routed = self.route(
            logits, candidates, fusion_prediction, temperature, blend
        )
        hard_index = logits.argmax(dim=1)
        hard_prediction = candidates.gather(1, hard_index.view(-1, 1))
        output.update({
            'fusion_feature': self._fusion_feature,
            'candidate_logits': logits,
            'candidate_values': candidates,
            'candidate_probabilities': probabilities,
            'routed_prediction': routed,
            'hard_candidate_index': hard_index,
            'hard_prediction': hard_prediction,
        })
        return output
