"""Fixed train-only Label Density Smoothing (LDS-v1) helpers.

No loader is constructed here: callers supply immutable train labels in
dataset-index order, keeping LDS independent of shuffling and held-out labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from .missing_utils import MODALITY_MASKS, compute_full_dlf_loss, compute_task_loss, mode_to_mask, regression_metrics


LDS_METHOD_NAME = "DLF-LDS-ModDrop-v1"
LDS_VERSION = "LDS-v1"
LDS_BIN_COUNT = 60
LDS_LABEL_MIN, LDS_LABEL_MAX = -3.0, 3.0
LDS_KERNEL_RADIUS, LDS_KERNEL_SIGMA = 2, 1.0
LDS_ALPHA, LDS_EPSILON = 0.5, 1e-6
LDS_CLIP_MIN, LDS_CLIP_MAX, LDS_CLIP_MAX_FRACTION = 0.2, 5.0, 0.20


def lds_v1_config() -> Dict[str, object]:
    """The only permitted LDS configuration; it has no tuning CLI."""
    return {
        "method": LDS_METHOD_NAME,
        "version": LDS_VERSION,
        "label_min": LDS_LABEL_MIN,
        "label_max": LDS_LABEL_MAX,
        "bin_count": LDS_BIN_COUNT,
        "bin_edges": np.linspace(LDS_LABEL_MIN, LDS_LABEL_MAX, LDS_BIN_COUNT + 1).tolist(),
        "kernel": "gaussian",
        "kernel_radius": LDS_KERNEL_RADIUS,
        "kernel_size": 2 * LDS_KERNEL_RADIUS + 1,
        "kernel_sigma": LDS_KERNEL_SIGMA,
        "alpha": LDS_ALPHA,
        "epsilon": LDS_EPSILON,
        "clip_min": LDS_CLIP_MIN,
        "clip_max": LDS_CLIP_MAX,
        "post_clip_normalize_mean": 1.0,
        "sampling": "unchanged_stage1_shuffle",
        "weight_source": "train_labels_only",
    }


def assert_fixed_lds_config(config: Dict[str, object]) -> None:
    if config != lds_v1_config():
        raise ValueError("LDS-v1 protocol configuration mismatch; tuning is forbidden.")


def config_sha256(config: Dict[str, object] | None = None) -> str:
    config = lds_v1_config() if config is None else config
    assert_fixed_lds_config(config)
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _labels(labels: Iterable[float]) -> np.ndarray:
    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("LDS labels must be non-empty and finite.")
    return values


def label_to_bin_indices(labels: Iterable[float]) -> np.ndarray:
    """Exact 60 bins: left closed/right open, except +3.0 in final bin."""
    values = _labels(labels)
    if np.any(values < LDS_LABEL_MIN) or np.any(values > LDS_LABEL_MAX):
        raise ValueError("LDS labels outside [-3.0, 3.0] are forbidden; no clipping is allowed.")
    width = (LDS_LABEL_MAX - LDS_LABEL_MIN) / LDS_BIN_COUNT
    result = np.floor((values - LDS_LABEL_MIN) / width).astype(np.int64)
    result[values == LDS_LABEL_MAX] = LDS_BIN_COUNT - 1
    if np.any(result < 0) or np.any(result >= LDS_BIN_COUNT):
        raise AssertionError("LDS bin assignment escaped its fixed range.")
    return result


def gaussian_kernel() -> np.ndarray:
    x = np.arange(-LDS_KERNEL_RADIUS, LDS_KERNEL_RADIUS + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * LDS_KERNEL_SIGMA ** 2))
    return kernel / kernel.sum()


@dataclass(frozen=True)
class LDSArtifacts:
    labels: np.ndarray
    bin_indices: np.ndarray
    counts: np.ndarray
    smooth_counts: np.ndarray
    kernel: np.ndarray
    raw_weights_by_bin: np.ndarray
    normalized_weights_by_bin: np.ndarray
    clipped_weights_by_bin: np.ndarray
    final_weights_by_bin: np.ndarray
    sample_weights: np.ndarray


def prepare_train_label_weights(train_labels: Iterable[float]) -> LDSArtifacts:
    """Compute fixed weights from train labels only, preserving index binding."""
    labels = _labels(train_labels)
    bins = label_to_bin_indices(labels)
    counts = np.bincount(bins, minlength=LDS_BIN_COUNT).astype(np.int64)
    kernel = gaussian_kernel()
    smooth = np.convolve(counts.astype(np.float64), kernel, mode="same")
    if not np.all(np.isfinite(smooth)) or np.any(smooth < 0.0):
        raise FloatingPointError("Invalid LDS density.")
    raw = (smooth + LDS_EPSILON) ** (-LDS_ALPHA)
    normalized = raw / raw[bins].mean()
    clipped = np.clip(normalized, LDS_CLIP_MIN, LDS_CLIP_MAX)
    final = clipped / clipped[bins].mean()
    artifacts = LDSArtifacts(labels, bins, counts, smooth, kernel, raw, normalized, clipped, final, final[bins])
    validate_lds_artifacts(artifacts)
    return artifacts


def validate_lds_artifacts(artifacts: LDSArtifacts) -> None:
    if int(artifacts.counts.sum()) != len(artifacts.labels):
        raise AssertionError("LDS counts do not cover each train label exactly once.")
    if not np.all(np.isfinite(artifacts.smooth_counts)) or np.any(artifacts.smooth_counts < 0.0):
        raise AssertionError("LDS densities must be finite and non-negative.")
    weights = artifacts.sample_weights
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise AssertionError("LDS weights must be finite and positive.")
    if abs(float(weights.mean()) - 1.0) > 1e-6 or float(weights.max()) > 5.5:
        raise AssertionError("LDS final weight normalization or maximum is invalid.")
    if len(np.unique(np.round(weights, 12))) < 3:
        raise AssertionError("LDS requires at least three unique sample weights.")
    clipped_max = artifacts.normalized_weights_by_bin[artifacts.bin_indices] >= LDS_CLIP_MAX
    if float(clipped_max.mean()) > LDS_CLIP_MAX_FRACTION:
        raise RuntimeError("More than 20% of samples hit the LDS maximum clip; stop without adjustment.")
    order = np.argsort(artifacts.smooth_counts)
    if np.any(np.diff(artifacts.final_weights_by_bin[order]) > 1e-12):
        raise AssertionError("Lower density must have non-decreasing LDS weight.")


def sample_weights_for_indices(artifacts: LDSArtifacts, indices: torch.Tensor, device=None) -> torch.Tensor:
    """Gather weights by original sample index, never by shuffled batch position."""
    array = indices.detach().view(-1).cpu().numpy().astype(np.int64)
    if np.any(array < 0) or np.any(array >= len(artifacts.sample_weights)):
        raise IndexError("Out-of-range dataset index in LDS training batch.")
    return torch.as_tensor(artifacts.sample_weights[array].astype(np.float32), device=device or indices.device)


def compute_task_loss_per_sample(output: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    """The original five task heads with exact factors 1,1,3,1,1."""
    target = labels.view(-1, 1)
    values = (
        torch.abs(output["output_logit"] - target)
        + torch.abs(output["logits_c"] - target)
        + 3.0 * torch.abs(output["logits_l_hetero"] - target)
        + torch.abs(output["logits_v_hetero"] - target)
        + torch.abs(output["logits_a_hetero"] - target)
    )
    return values.reshape(-1)


def weighted_task_loss(per_sample_loss: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    losses = per_sample_loss.view(-1)
    weights = sample_weights.to(device=losses.device, dtype=losses.dtype).view(-1)
    if losses.shape != weights.shape:
        raise ValueError("LDS loss and weight shapes must match.")
    if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
        raise FloatingPointError("Invalid LDS batch weights.")
    return (weights * losses).sum() / weights.sum()


def compute_full_dlf_loss_lds(output, labels, criterion, cosine, sim_loss, sample_weights=None):
    """Preserve Stage 1 full loss and replace only its task reduction."""
    total, details = compute_full_dlf_loss(output, labels, criterion, cosine, sim_loss)
    if sample_weights is None:
        return total, details
    weighted = weighted_task_loss(compute_task_loss_per_sample(output, labels), sample_weights)
    revised = dict(details)
    revised["task_loss_unweighted"] = details["task_loss"]
    revised["task_loss"] = weighted
    return total - details["task_loss"] + weighted, revised


def compute_missing_task_loss_lds(output, labels, sample_weights, criterion):
    """Missing view has only the LDS weighted five-head task loss."""
    weighted = weighted_task_loss(compute_task_loss_per_sample(output, labels), sample_weights)
    unweighted, _ = compute_task_loss(output, labels, criterion)
    return weighted, unweighted


FIXED_EMOTION_BINS = ("[-3,-1)", "[-1,0)", "{0}", "(0,1]", "(1,3]")


def fixed_emotion_bin_masks(labels: Iterable[float]) -> Dict[str, np.ndarray]:
    values = _labels(labels)
    return {
        "[-3,-1)": (values >= -3.0) & (values < -1.0),
        "[-1,0)": (values >= -1.0) & (values < 0.0),
        "{0}": values == 0.0,
        "(0,1]": (values > 0.0) & (values <= 1.0),
        "(1,3]": (values > 1.0) & (values <= 3.0),
    }


def density_group_thresholds(train_weights: np.ndarray) -> Tuple[float, float, float]:
    return tuple(float(v) for v in np.quantile(np.asarray(train_weights), [0.25, 0.50, 0.75]))


def assign_density_groups(weights: np.ndarray, thresholds: Tuple[float, float, float]) -> np.ndarray:
    q25, q50, q75 = thresholds
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    return np.select([values <= q25, values <= q50, values <= q75], ["head", "middle_low", "middle_high"], default="tail")


def validation_diagnostics(predictions, labels, artifacts: LDSArtifacts, seed: int, epoch: int):
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = _labels(labels)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and label counts differ.")
    bin_rows, nonempty_mae = [], []
    for name, mask in fixed_emotion_bin_masks(target).items():
        count = int(mask.sum())
        if count:
            errors = prediction[mask] - target[mask]
            mae, mean_pred, mean_label, bias = float(np.abs(errors).mean()), float(prediction[mask].mean()), float(target[mask].mean()), float(errors.mean())
            nonempty_mae.append(mae)
        else:
            mae = mean_pred = mean_label = bias = float("nan")
        bin_rows.append({"Seed": seed, "Epoch": epoch, "Bin": name, "Count": count, "MAE": mae, "MeanPred": mean_pred, "MeanLabel": mean_label, "Bias": bias})
    macro = float(np.mean(nonempty_mae)) if nonempty_mae else float("nan")
    for row in bin_rows:
        row["MacroBinMAE"] = macro
    valid_weights = artifacts.final_weights_by_bin[label_to_bin_indices(target)]
    thresholds = density_group_thresholds(artifacts.sample_weights)
    groups, group_rows = assign_density_groups(valid_weights, thresholds), []
    for name in ("head", "middle_low", "middle_high", "tail"):
        mask, count = groups == name, int((groups == name).sum())
        if count:
            metrics = regression_metrics(torch.as_tensor(prediction[mask], dtype=torch.float32), torch.as_tensor(target[mask], dtype=torch.float32))
            mae, corr, acc2, bias = metrics["MAE"], metrics["Corr"], metrics["acc_2"], float((prediction[mask] - target[mask]).mean())
        else:
            mae = corr = acc2 = bias = float("nan")
        group_rows.append({"Seed": seed, "Epoch": epoch, "Group": name, "Count": count, "MAE": mae, "Corr": corr, "Acc2": acc2, "Bias": bias, "Q25": thresholds[0], "Q50": thresholds[1], "Q75": thresholds[2]})
    return bin_rows, group_rows, macro


def evaluate_lds_validation(model, dataloader, device, criterion, artifacts, seed, epoch):
    """Validation-only LAV/LA/LV/L metrics; all losses here remain unweighted."""
    model.eval()
    collected = {mode: {"pred": [], "label": [], "loss": []} for mode in MODALITY_MASKS}
    with torch.no_grad():
        for batch in dataloader:
            text, audio, vision = batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device)
            labels = batch["labels"]["M"].to(device).view(-1, 1)
            for mode in MODALITY_MASKS:
                output = model(text, audio, vision, mode_to_mask(mode, labels.size(0), device, audio.dtype))
                collected[mode]["pred"].append(output["output_logit"].detach().cpu())
                collected[mode]["label"].append(labels.detach().cpu())
                collected[mode]["loss"].append(criterion(output["output_logit"], labels).item())
    metrics_by_mode, bin_rows, group_rows = {}, [], []
    for mode, values in collected.items():
        prediction, labels = torch.cat(values["pred"]), torch.cat(values["label"])
        metrics = regression_metrics(prediction, labels)
        metrics["Loss"] = float(np.mean(values["loss"]))
        mode_bins, mode_groups, macro = validation_diagnostics(prediction.numpy(), labels.numpy(), artifacts, seed, epoch)
        for row in mode_bins:
            row["Mode"] = mode
        for row in mode_groups:
            row["Mode"] = mode
        metrics["MacroBinMAE"] = macro
        metrics_by_mode[mode] = metrics
        bin_rows.extend(mode_bins)
        group_rows.extend(mode_groups)
    return metrics_by_mode, bin_rows, group_rows


def lds_checkpoint_path(root, dataset_name: str, seed: int) -> Path:
    return Path(root) / "missing_baseline" / "lds_moddrop_v1" / "DLF_{}_seed{}_best.pth".format(dataset_name, seed)


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_density_audit(artifacts: LDSArtifacts, output_dir) -> Dict[str, object]:
    """Required audit output; caller supplies only training-label artifacts."""
    output_dir = Path(output_dir)
    edges, rows = np.linspace(-3.0, 3.0, 61), []
    for i in range(LDS_BIN_COUNT):
        rows.append({"bin_index": i, "left_edge": float(edges[i]), "right_edge": float(edges[i + 1]), "count": int(artifacts.counts[i]), "smooth_count": float(artifacts.smooth_counts[i]), "raw_weight": float(artifacts.raw_weights_by_bin[i]), "normalized_weight": float(artifacts.normalized_weights_by_bin[i]), "final_weight": float(artifacts.final_weights_by_bin[i])})
    _write_csv(output_dir / "train_label_density_bins.csv", ("bin_index", "left_edge", "right_edge", "count", "smooth_count", "raw_weight", "normalized_weight", "final_weight"), rows)
    fixed = [{"bin": name, "count": int(mask.sum()), "average_weight": float(artifacts.sample_weights[mask].mean()) if mask.any() else float("nan")} for name, mask in fixed_emotion_bin_masks(artifacts.labels).items()]
    _write_csv(output_dir / "train_fixed_emotion_bins.csv", ("bin", "count", "average_weight"), fixed)
    weights = artifacts.sample_weights
    summary = {
        "method": LDS_METHOD_NAME, "config_sha256": config_sha256(), "train_sample_count": int(len(artifacts.labels)),
        "label_min": float(artifacts.labels.min()), "label_max": float(artifacts.labels.max()), "label_mean": float(artifacts.labels.mean()), "label_std": float(artifacts.labels.std()), "nonempty_bins": int(np.count_nonzero(artifacts.counts)),
        "weight_percentiles": {name: float(np.percentile(weights, value)) for name, value in (("min", 0), ("p1", 1), ("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90), ("p99", 99), ("max", 100))},
        "weight_mean": float(weights.mean()), "weight_std": float(weights.std()),
        "fraction_clipped_min": float((artifacts.normalized_weights_by_bin[artifacts.bin_indices] <= LDS_CLIP_MIN).mean()), "fraction_clipped_max": float((artifacts.normalized_weights_by_bin[artifacts.bin_indices] >= LDS_CLIP_MAX).mean()), "fixed_emotion_bins": fixed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_sample_weights_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    (output_dir / "LDS_V1_CONFIG.json").write_text(json.dumps(lds_v1_config(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
