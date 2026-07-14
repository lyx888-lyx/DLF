"""Utilities shared by the Stage 1 missing-modality baselines.

The modality order is permanently [text, audio, vision]. Text is always
present in Stage 1, so the only sampled training modes are LA, LV, and L.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .HingeLoss import HingeLoss


MODALITY_MASKS = {
    "LAV": (1.0, 1.0, 1.0),
    "LA": (1.0, 1.0, 0.0),
    "LV": (1.0, 0.0, 1.0),
    "L": (1.0, 0.0, 0.0),
}
MISSING_MODES = ("LA", "LV", "L")
TRAINING_SPLITS = ("train", "valid")


def mode_to_mask(mode, batch_size=None, device=None, dtype=torch.float32):
    """Return a modality-present mask in the fixed [text, audio, vision] order."""
    if mode not in MODALITY_MASKS:
        raise ValueError("Unsupported Stage 1 modality mode: {}".format(mode))
    mask = torch.tensor(MODALITY_MASKS[mode], dtype=dtype, device=device)
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(int(batch_size), -1)
    return mask


def sample_missing_masks(batch_size, generator, device=None, dtype=torch.float32):
    """Sample LA/LV/L independently for each sample using the supplied RNG."""
    if generator is None:
        raise ValueError("A dedicated, seeded generator is required for ModDrop.")
    choices = torch.randint(len(MISSING_MODES), (int(batch_size),), generator=generator)
    candidates = torch.stack([mode_to_mask(mode) for mode in MISSING_MODES])
    return candidates.index_select(0, choices).to(device=device, dtype=dtype)


def count_missing_modes(modality_mask):
    """Return the realized LA/LV/L counts for an independently sampled batch."""
    result = Counter({mode: 0 for mode in MISSING_MODES})
    for row in modality_mask.detach().cpu():
        values = tuple(int(value.item()) for value in row)
        for mode, expected in MODALITY_MASKS.items():
            if values == tuple(int(value) for value in expected):
                if mode not in MISSING_MODES:
                    raise ValueError("ModDrop may not sample {}.".format(mode))
                result[mode] += 1
                break
        else:
            raise ValueError("Unknown modality mask {}".format(values))
    return dict(result)


def _present_column(modality_mask, modality_index, values):
    if modality_mask.ndim != 2 or modality_mask.size(1) != 3:
        raise ValueError("modality_mask must have shape [batch, 3].")
    if modality_mask.size(0) != values.size(0):
        raise ValueError("modality_mask batch size does not match model inputs.")
    shape = [values.size(0)] + [1] * (values.ndim - 1)
    return modality_mask[:, modality_index].to(values).view(*shape)


def apply_direct_mask(audio, vision, modality_mask):
    """Pure input-zeroing used by DLF-DirectMask; it never uses missing tokens."""
    audio_present = _present_column(modality_mask, 1, audio)
    vision_present = _present_column(modality_mask, 2, vision)
    return audio * audio_present, vision * vision_present


def apply_moddrop_tokens(audio, vision, modality_mask, missing_audio_token, missing_vision_token):
    """Replace whole missing audio/vision modalities with learnable tokens."""
    if missing_audio_token.shape != (1, 1, audio.size(-1)):
        raise ValueError("missing_audio_token has incompatible feature dimension.")
    if missing_vision_token.shape != (1, 1, vision.size(-1)):
        raise ValueError("missing_vision_token has incompatible feature dimension.")
    audio_present = _present_column(modality_mask, 1, audio)
    vision_present = _present_column(modality_mask, 2, vision)
    masked_audio = audio_present * audio + (1.0 - audio_present) * missing_audio_token
    masked_vision = vision_present * vision + (1.0 - vision_present) * missing_vision_token
    return masked_audio, masked_vision


class MissingModalityWrapper(nn.Module):
    """Add Stage 1 ModDrop inputs and a zero-initialized mask residual to DLF."""

    def __init__(self, backbone, audio_dim, vision_dim):
        super().__init__()
        if not hasattr(backbone, "out_layer") or not hasattr(backbone.out_layer, "in_features"):
            raise ValueError("The DLF backbone must expose its final fusion dimension.")
        self.backbone = backbone
        self.missing_audio_token = nn.Parameter(torch.zeros(1, 1, int(audio_dim)))
        self.missing_vision_token = nn.Parameter(torch.zeros(1, 1, int(vision_dim)))
        self.mask_adapter = nn.Linear(3, backbone.out_layer.in_features, bias=False)
        nn.init.zeros_(self.mask_adapter.weight)

    def forward(self, text, audio, vision, modality_mask):
        modality_mask = modality_mask.to(device=audio.device, dtype=audio.dtype)
        masked_audio, masked_vision = apply_moddrop_tokens(
            audio, vision, modality_mask, self.missing_audio_token, self.missing_vision_token
        )
        missing_indicator = 1.0 - modality_mask
        fusion_residual = self.mask_adapter(missing_indicator)
        return self.backbone(text, masked_audio, masked_vision, fusion_residual=fusion_residual)


def compute_task_loss(output, labels, criterion):
    """Reuse the original DLF five-head task-loss weighting exactly."""
    components = {
        "task_all": criterion(output["output_logit"], labels),
        "task_c": criterion(output["logits_c"], labels),
        "task_l_hetero": criterion(output["logits_l_hetero"], labels),
        "task_v_hetero": criterion(output["logits_v_hetero"], labels),
        "task_a_hetero": criterion(output["logits_a_hetero"], labels),
    }
    task_loss = (
        components["task_all"]
        + components["task_c"]
        + 3.0 * components["task_l_hetero"]
        + components["task_v_hetero"]
        + components["task_a_hetero"]
    )
    return task_loss, components


def compute_full_dlf_loss(output, labels, criterion, cosine=None, sim_loss=None):
    """Original DLF combined loss for the complete LAV training view only."""
    cosine = cosine if cosine is not None else nn.CosineEmbeddingLoss()
    sim_loss = sim_loss if sim_loss is not None else HingeLoss()
    task_loss, task_components = compute_task_loss(output, labels, criterion)

    reconstruction_loss = (
        F.mse_loss(output["recon_l"], output["origin_l"])
        + F.mse_loss(output["recon_v"], output["origin_v"])
        + F.mse_loss(output["recon_a"], output["origin_a"])
    )
    specific_loss = (
        F.mse_loss(output["s_l"].permute(1, 2, 0), output["s_l_r"])
        + F.mse_loss(output["s_v"].permute(1, 2, 0), output["s_v_r"])
        + F.mse_loss(output["s_a"].permute(1, 2, 0), output["s_a_r"])
    )

    feature_dim = output["s_l"].shape[-1]
    target = torch.full(
        (output["s_l"].reshape(-1, feature_dim).size(0),),
        -1.0,
        dtype=output["s_l"].dtype,
        device=output["s_l"].device,
    )
    orthogonality_loss = (
        cosine(output["s_l"].reshape(-1, feature_dim), output["c_l"].reshape(-1, feature_dim), target)
        + cosine(output["s_v"].reshape(-1, feature_dim), output["c_v"].reshape(-1, feature_dim), target)
        + cosine(output["s_a"].reshape(-1, feature_dim), output["c_a"].reshape(-1, feature_dim), target)
    )

    shared_features = (output["c_l_sim"], output["c_v_sim"], output["c_a_sim"])
    features = torch.cat(
        [feature[index].view(1, -1) for index in range(labels.size(0)) for feature in shared_features],
        dim=0,
    )
    ids = torch.cat([labels[index].view(1, -1).repeat(3, 1) for index in range(labels.size(0))], dim=0)
    similarity_loss = sim_loss(ids, features)

    total_loss = task_loss + (specific_loss + reconstruction_loss + 0.1 * (similarity_loss + orthogonality_loss)) * 0.1
    details = {
        "task_loss": task_loss,
        "reconstruction_loss": reconstruction_loss,
        "specific_loss": specific_loss,
        "orthogonality_loss": orthogonality_loss,
        "similarity_loss": similarity_loss,
    }
    details.update(task_components)
    return total_loss, details


def regression_metrics(predictions, labels):
    """MOSI regression metrics in raw [0, 1] scale for CSV output."""
    prediction = predictions.detach().view(-1).cpu().numpy()
    target = labels.detach().view(-1).cpu().numpy()
    clipped_7_prediction = np.clip(prediction, -3.0, 3.0)
    clipped_7_target = np.clip(target, -3.0, 3.0)
    clipped_5_prediction = np.clip(prediction, -2.0, 2.0)
    clipped_5_target = np.clip(target, -2.0, 2.0)
    nonzero = target != 0
    if np.any(nonzero):
        binary_prediction = prediction[nonzero] > 0
        binary_target = target[nonzero] > 0
        acc_2 = float(np.mean(binary_prediction == binary_target))
        f1 = float(f1_score(binary_target, binary_prediction, average="weighted", zero_division=0))
    else:
        acc_2, f1 = 0.0, 0.0
    if prediction.size < 2 or np.std(prediction) == 0 or np.std(target) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "acc_7": float(np.mean(np.round(clipped_7_prediction) == np.round(clipped_7_target))),
        "acc_5": float(np.mean(np.round(clipped_5_prediction) == np.round(clipped_5_target))),
        "acc_2": acc_2,
        "F1_score": f1,
        "Corr": corr,
        "MAE": float(np.mean(np.abs(prediction - target))),
    }


def evaluate_all_modes(model, dataloader, device, method, criterion):
    """Evaluate LAV/LA/LV/L on one validation or explicitly confirmed test loader."""
    if method not in ("directmask", "moddrop"):
        raise ValueError("Unknown method: {}".format(method))
    model.eval()
    collected = {mode: {"pred": [], "label": [], "loss": []} for mode in MODALITY_MASKS}
    with torch.no_grad():
        for batch_data in dataloader:
            text = batch_data["text"].to(device)
            audio = batch_data["audio"].to(device)
            vision = batch_data["vision"].to(device)
            labels = batch_data["labels"]["M"].to(device).view(-1, 1)
            for mode in MODALITY_MASKS:
                mask = mode_to_mask(mode, batch_size=labels.size(0), device=device, dtype=audio.dtype)
                if method == "directmask":
                    input_audio, input_vision = apply_direct_mask(audio, vision, mask)
                    output = model(text, input_audio, input_vision)
                else:
                    output = model(text, audio, vision, mask)
                collected[mode]["pred"].append(output["output_logit"].detach().cpu())
                collected[mode]["label"].append(labels.detach().cpu())
                collected[mode]["loss"].append(criterion(output["output_logit"], labels).item())
    results = {}
    for mode, values in collected.items():
        metrics = regression_metrics(torch.cat(values["pred"]), torch.cat(values["label"]))
        metrics["Loss"] = float(np.mean(values["loss"]))
        results[mode] = metrics
    return results


def validation_objective(metrics_by_mode):
    """The required validation-only checkpoint criterion (lower is better)."""
    return 0.5 * metrics_by_mode["LAV"]["MAE"] + 0.5 * np.mean(
        [metrics_by_mode[mode]["MAE"] for mode in MISSING_MODES]
    )


def flatten_mode_metrics(metrics_by_mode):
    result = {}
    for mode, metrics in metrics_by_mode.items():
        for key, value in metrics.items():
            result["{}_{}".format(mode, key)] = float(value)
    return result


def write_result_csvs(rows, output_dir, dataset_name):
    """Write per-seed and raw-scale mean/std CSV files without altering Gate 3 results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "{}_per_seed.csv".format(dataset_name), index=False)
    numeric = frame.select_dtypes(include=[np.number])
    summary = pd.DataFrame(
        {"Metric": numeric.columns, "Mean": numeric.mean().values, "Std": numeric.std(ddof=0).values}
    )
    summary.to_csv(output_dir / "{}_summary.csv".format(dataset_name), index=False)


def missing_checkpoint_path(root, dataset_name, seed):
    return Path(root) / "missing_baseline" / "moddrop" / "DLF_{}_seed{}_best.pth".format(dataset_name, seed)


def clean_checkpoint_path(root, dataset_name, seed):
    return Path(root) / "DLF_{}_seed{}_best.pth".format(dataset_name, seed)


def build_single_split_loader(args, split, num_workers):
    """Build exactly one non-shuffled split; callers must gate test explicitly."""
    if split not in ("train", "valid", "test"):
        raise ValueError("Unknown split: {}".format(split))
    from data_loader import MMDataset

    dataset = MMDataset(args, mode=split)
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=False)
