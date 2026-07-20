"""Stage 20 SAFE-DLF primitives.

The module is deliberately train/official-valid only.  It does not construct a
dataset and contains no test-evaluation entry point.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from .missing_utils import apply_direct_mask, apply_moddrop_tokens


METHODS = ("sao_pm", "safe_full")
ALLOWED_SPLITS = ("train", "valid")
EPSILON = 1e-12


def require_train_or_valid(split):
    if split not in ALLOWED_SPLITS:
        raise RuntimeError("Stage 20 permits train/official-valid only.")
    return split


def _present_column(modality_mask, index, value):
    if modality_mask.ndim != 2 or modality_mask.size(1) != 3:
        raise ValueError("modality_mask must have shape [batch, 3].")
    if modality_mask.size(0) != value.size(0):
        raise ValueError("modality_mask batch size mismatch.")
    return modality_mask[:, index].to(value).view(
        value.size(0), *([1] * (value.ndim - 1))
    )


class SafeMissingModalityWrapper(nn.Module):
    """DLF wrapper with hard per-sample availability projection.

    Parameter names intentionally match ``MissingModalityWrapper`` so a frozen
    Stage 19 Uniform checkpoint can be loaded without translation.
    """

    def __init__(self, backbone, audio_dim, vision_dim):
        super().__init__()
        self.backbone = backbone
        self.missing_audio_token = nn.Parameter(
            torch.zeros(1, 1, int(audio_dim))
        )
        self.missing_vision_token = nn.Parameter(
            torch.zeros(1, 1, int(vision_dim))
        )
        self.mask_adapter = nn.Linear(
            3, backbone.out_layer.in_features, bias=False
        )
        nn.init.zeros_(self.mask_adapter.weight)

    def forward(self, text, audio, vision, modality_mask):
        modality_mask = modality_mask.to(device=audio.device, dtype=audio.dtype)
        if modality_mask.ndim != 2 or modality_mask.shape != (audio.size(0), 3):
            raise ValueError("modality_mask must have shape [batch, 3].")
        if not torch.all(modality_mask[:, 0] == 1):
            raise ValueError("Language must be present.")
        missing_indicator = 1.0 - modality_mask
        fusion_residual = self.mask_adapter(missing_indicator)

        # The complete-modality path is the exact original graph.  This is
        # stronger than merely obtaining close outputs and protects LAV parity.
        if bool(torch.all(modality_mask == 1)):
            return self.backbone(
                text, audio, vision, fusion_residual=fusion_residual
            )

        masked_audio, masked_vision = apply_direct_mask(
            audio, vision, modality_mask
        )
        return self.backbone(
            text,
            masked_audio,
            masked_vision,
            fusion_residual=fusion_residual,
            availability_mask=modality_mask,
        )


def support_aligned_task_loss(output, labels, modality_mask):
    """Original five-head weighting with absent heads set to zero.

    Each active head keeps its original coefficient and the reduction remains a
    mean over the full batch.  There is intentionally no active-term
    normalization.
    """

    labels = labels.view(-1, 1)
    audio_present = _present_column(
        modality_mask, 1, output["logits_a_hetero"]
    )
    vision_present = _present_column(
        modality_mask, 2, output["logits_v_hetero"]
    )
    absolute = lambda prediction: torch.abs(prediction - labels)
    components = OrderedDict(
        task_all=absolute(output["output_logit"]).mean(),
        task_c=absolute(output["logits_c"]).mean(),
        task_l_hetero=absolute(output["logits_l_hetero"]).mean(),
        task_v_hetero=(
            absolute(output["logits_v_hetero"]) * vision_present
        ).mean(),
        task_a_hetero=(
            absolute(output["logits_a_hetero"]) * audio_present
        ).mean(),
    )
    loss = (
        components["task_all"]
        + components["task_c"]
        + 3.0 * components["task_l_hetero"]
        + components["task_v_hetero"]
        + components["task_a_hetero"]
    )
    zero = output["output_logit"].sum() * 0.0
    components.update(
        absent_reconstruction=zero,
        absent_specific_consistency=zero,
        absent_orthogonality=zero,
        absent_triplet=zero,
    )
    return loss, components


def supported_triplet_inputs(labels, output, modality_mask):
    """Build a triplet pool containing present embeddings only."""

    labels = labels.view(-1, 1)
    shared = (
        output["c_l_sim"],
        output["c_a_sim"],
        output["c_v_sim"],
    )
    features = []
    ids = []
    support = []
    for sample in range(labels.size(0)):
        for modality, feature in enumerate(shared):
            if bool(modality_mask[sample, modality] > 0.5):
                features.append(feature[sample].view(1, -1))
                ids.append(labels[sample].view(1, -1))
                support.append((sample, modality))
    if not features:
        raise RuntimeError("The supported triplet pool cannot be empty.")
    return torch.cat(ids), torch.cat(features), support


def unsupported_task_components(output, labels, modality_mask):
    """Return weighted absent-head terms used by the frozen Uniform objective."""

    labels = labels.view(-1, 1)
    audio_absent = 1.0 - _present_column(
        modality_mask, 1, output["logits_a_hetero"]
    )
    vision_absent = 1.0 - _present_column(
        modality_mask, 2, output["logits_v_hetero"]
    )
    return OrderedDict(
        absent_specific_prediction_audio=(
            torch.abs(output["logits_a_hetero"] - labels) * audio_absent
        ).mean(),
        absent_specific_prediction_vision=(
            torch.abs(output["logits_v_hetero"] - labels) * vision_absent
        ).mean(),
        absent_reconstruction=output["output_logit"].sum() * 0.0,
        absent_specific_consistency=output["output_logit"].sum() * 0.0,
        absent_orthogonality=output["output_logit"].sum() * 0.0,
        absent_triplet=output["output_logit"].sum() * 0.0,
    )


def filler_variants(audio, vision, generator=None):
    """Return deterministic zero/permutation/Gaussian/high fillers."""

    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(20)
    batch = audio.size(0)
    permutation = torch.arange(batch - 1, -1, -1, device=audio.device)
    audio_mean = audio.detach().float().mean(dim=(0, 1), keepdim=True)
    audio_std = audio.detach().float().std(dim=(0, 1), keepdim=True).clamp_min(
        1e-6
    )
    vision_mean = vision.detach().float().mean(dim=(0, 1), keepdim=True)
    vision_std = (
        vision.detach()
        .float()
        .std(dim=(0, 1), keepdim=True)
        .clamp_min(1e-6)
    )
    gaussian_audio = torch.randn(
        audio.shape, generator=generator, dtype=torch.float32
    ).to(audio.device)
    gaussian_vision = torch.randn(
        vision.shape, generator=generator, dtype=torch.float32
    ).to(vision.device)
    gaussian_audio = (
        gaussian_audio * audio_std.to(audio.device) + audio_mean.to(audio.device)
    ).to(audio)
    gaussian_vision = (
        gaussian_vision * vision_std.to(vision.device)
        + vision_mean.to(vision.device)
    ).to(vision)
    return OrderedDict(
        zero=(torch.zeros_like(audio), torch.zeros_like(vision)),
        permutation=(
            audio.index_select(0, permutation),
            vision.index_select(0, permutation),
        ),
        gaussian=(gaussian_audio, gaussian_vision),
        high=(
            torch.full_like(audio, 100.0),
            torch.full_like(vision, 100.0),
        ),
    )


def branch_parameter_groups(model):
    groups = {"audio": [], "vision": [], "language": [], "shared": []}
    for name, parameter in model.named_parameters():
        if any(token in name for token in ("_a", "audio")):
            groups["audio"].append(parameter)
        elif any(token in name for token in ("_v", "vision")):
            groups["vision"].append(parameter)
        elif any(token in name for token in ("_l", "text_model")):
            groups["language"].append(parameter)
        else:
            groups["shared"].append(parameter)
    return groups


def gradient_norm(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(
        loss,
        list(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    square = loss.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            square = square + gradient.float().square().sum()
    return float(torch.sqrt(square).detach().cpu())

