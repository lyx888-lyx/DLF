"""Train-only counterfactual-compatibility utilities for Stage 3B."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .fixed_kd_utils import (
    _restore_normal_position_cache,
    capture_rng_state,
    checkpoint_sha256,
    freeze_teacher,
    restore_rng_state,
)
from .missing_utils import MISSING_MODES, MissingModalityWrapper, mode_to_mask

CACHE_VERSION = "cf_compat_v1"
RANK_TRANSFORM = "(rank-0.5)/N"
CACHE_COLUMNS = (
    "sample_index", "sample_id", "label",
    "evaluator_LAV_pred", "evaluator_LA_pred", "evaluator_LV_pred", "evaluator_L_pred",
    "delta_LA", "delta_LV", "delta_L",
    "rank_LA", "rank_LV", "rank_L",
    "q_LA", "q_LV", "q_L",
    "compat_LA", "compat_LV", "compat_L",
)


def cache_paths(root, dataset):
    directory = Path(root) / "counterfactual_compatibility" / CACHE_VERSION / dataset
    return {
        "directory": directory,
        "csv": directory / "train_counterfactual_compatibility.csv",
        "config": directory / "cf_compat_config.json",
        "summary": directory / "cf_compat_summary.json",
        "bins": directory / "cf_compat_mode_bins.csv",
    }


def stable_average_ranks(values):
    """One-based average ranks with an explicitly stable NumPy mergesort."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Ranks require a non-empty finite vector.")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def compatibility_from_deltas(deltas):
    """Fixed independent per-mode empirical rank normalization."""
    delta = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if not np.isfinite(delta).all() or np.any(delta < 0):
        raise FloatingPointError("Deltas must be non-negative and finite.")
    rank = stable_average_ranks(delta)
    q = (rank - 0.5) / float(len(delta))
    compat = 1.0 - q
    if not (np.all((q > 0) & (q < 1)) and np.all((compat > 0) & (compat < 1))):
        raise AssertionError("q and compatibility must be strictly in (0,1).")
    return rank, q, compat


def modes_from_masks(mask):
    mapping = {(1, 1, 0): "LA", (1, 0, 1): "LV", (1, 0, 0): "L"}
    try:
        return [mapping[tuple(row)] for row in mask.detach().cpu().to(torch.int64).tolist()]
    except KeyError as error:
        raise ValueError("Only LA/LV/L may be sampled: {}".format(error))


def compatibility_for_modes(cache_by_index, indices, modes, device, dtype):
    if len(indices) != len(modes):
        raise ValueError("Indices and modes differ in length.")
    values = []
    for index, mode in zip(indices, modes):
        if mode not in MISSING_MODES or int(index) not in cache_by_index:
            raise KeyError("Invalid cache binding index={} mode={}".format(index, mode))
        values.append(float(cache_by_index[int(index)]["compat_{}".format(mode)]))
    result = torch.as_tensor(values, device=device, dtype=dtype)
    if not torch.isfinite(result).all() or torch.any(result <= 0) or torch.any(result >= 1):
        raise FloatingPointError("Compatibility must remain in (0,1).")
    return result


def stage3a_reliability(teacher_prediction, labels):
    prediction = teacher_prediction.detach().clone().view(-1)
    target = labels.detach().view(-1)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Reliability inputs must be finite.")
    reliability = torch.exp(-torch.abs(prediction - target))
    if not torch.isfinite(reliability).all() or torch.any(reliability <= 0) or torch.any(reliability > 1):
        raise FloatingPointError("Reliability must be in (0,1].")
    return reliability


def gate_weights(compatibility, teacher_prediction=None, labels=None, gate_mode="compat"):
    compatibility = compatibility.detach().view(-1)
    if gate_mode == "compat":
        reliability = torch.ones_like(compatibility)
        gate = compatibility
    elif gate_mode == "reliability_compat":
        if teacher_prediction is None or labels is None:
            raise ValueError("R-times-C requires teacher prediction and label.")
        reliability = stage3a_reliability(teacher_prediction, labels).to(compatibility)
        gate = reliability * compatibility
    else:
        raise ValueError("Unknown gate_mode: {}".format(gate_mode))
    if not torch.isfinite(gate).all() or torch.any(gate <= 0) or torch.any(gate > 1):
        raise FloatingPointError("Gate must be in (0,1].")
    return gate.detach(), reliability.detach()


def gated_kd_loss(student_prediction, teacher_prediction, gate):
    """SmoothL1 output KD; the gate is detached and only appears here."""
    student = student_prediction.view(-1)
    teacher = teacher_prediction.detach().clone().view(-1)
    weight = gate.detach().view(-1).to(student)
    each = torch.nn.functional.smooth_l1_loss(student, teacher, reduction="none")
    if each.shape != weight.shape:
        raise ValueError("KD and gate shapes differ.")
    total = torch.sum(weight * each) / (torch.sum(weight) + 1e-8)
    if not torch.isfinite(total):
        raise FloatingPointError("Gated KD is non-finite.")
    return total, each


def distribution_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Statistics require finite non-empty values.")
    q = np.quantile(values, [.1, .25, .5, .75, .9, .95])
    return dict(mean=float(values.mean()), std=float(values.std()), min=float(values.min()),
                p10=float(q[0]), p25=float(q[1]), median=float(q[2]), p75=float(q[3]),
                p90=float(q[4]), p95=float(q[5]), max=float(values.max()))


def effective_sample_size(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("ESS requires finite non-empty gates.")
    return float(values.sum() ** 2 / np.square(values).sum())


def _pearson(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if len(first) < 2 or first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def cache_summary(frame):
    result = {}
    for mode in MISSING_MODES:
        delta = frame["delta_{}".format(mode)].to_numpy(dtype=np.float64)
        compat = frame["compat_{}".format(mode)].to_numpy(dtype=np.float64)
        result[mode] = {
            "delta": distribution_stats(delta), "compat": distribution_stats(compat),
            "nonfinite_count": int((~np.isfinite(delta)).sum() + (~np.isfinite(compat)).sum()),
            "unique_delta_count": int(np.unique(delta).size),
            "tie_fraction": float(1 - np.unique(delta).size / len(delta)),
            "corr_delta_compat_pearson": _pearson(delta, compat),
            "corr_delta_compat_spearman": _pearson(stable_average_ranks(delta), stable_average_ranks(compat)),
        }
    return result


def cache_bin_rows(frame):
    rows = []
    for mode in MISSING_MODES:
        values = frame[["delta_{}".format(mode), "compat_{}".format(mode)]].copy()
        values.columns = ["delta", "compat"]
        values["quartile"] = pd.qcut(values.compat, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates="drop")
        for group, local in values.groupby("quartile", observed=False):
            if len(local):
                rows.append(dict(mode=mode, compat_quartile=str(group), count=int(len(local)),
                                 mean_delta=float(local.delta.mean()), mean_compat=float(local.compat.mean())))
    return rows


def locate_stage1_evaluator(result_root, dataset, seed):
    """Read the Stage 1 CSV recorded checkpoint and best epoch; never infer paths."""
    source = Path(result_root) / "missing_baseline" / "moddrop" / "train" / "{}_per_seed.csv".format(dataset)
    if not source.is_file():
        raise FileNotFoundError("Required Stage 1 result CSV absent: {}".format(source))
    rows = pd.read_csv(source)
    selected = rows.loc[rows.Seed.astype(int) == int(seed)]
    if len(selected) != 1 or "Checkpoint" not in selected:
        raise ValueError("Stage 1 CSV has no unique checkpoint for seed {}.".format(seed))
    checkpoint = Path(str(selected.iloc[0].Checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError("Stage 1 CSV checkpoint is absent: {}".format(checkpoint))
    return checkpoint, int(selected.iloc[0].BestEpoch), source


def build_frozen_evaluator(model_factory, args, checkpoint):
    """Create a distinct frozen Stage 1 ModDrop evaluator without RNG drift."""
    state = capture_rng_state()
    try:
        backbone = model_factory(args).to(args.device)
        evaluator = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
        evaluator.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
        return freeze_teacher(evaluator)
    finally:
        restore_rng_state(state)


def evaluator_prediction(evaluator, text, audio, vision, mode):
    mask = mode_to_mask(mode, text.size(0), audio.device, audio.dtype)
    with torch.inference_mode():
        prediction = evaluator(text, audio, vision, mask)["output_logit"]
    normal = prediction.detach().clone()
    _restore_normal_position_cache()
    return normal


@contextmanager
def preserve_rng(generator):
    state = capture_rng_state()
    generator_state = generator.get_state().clone() if generator is not None else None
    try:
        yield
    finally:
        restore_rng_state(state)
        if generator is not None:
            generator.set_state(generator_state)


def build_counterfactual_cache(evaluator, train_loader, device, missing_generator):
    """Make the single train-only LAV/LA/LV/L evaluator pass."""
    records = []
    with preserve_rng(missing_generator):
        evaluator.eval()
        for batch in train_loader:
            text, audio, vision = batch["text"].to(device), batch["audio"].to(device), batch["vision"].to(device)
            labels = batch["labels"]["M"].view(-1).cpu().numpy()
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            identifiers = list(batch["id"])
            predictions = {mode: evaluator_prediction(evaluator, text, audio, vision, mode).view(-1).cpu().numpy()
                           for mode in ("LAV",) + MISSING_MODES}
            for position, index in enumerate(indices):
                records.append(dict(sample_index=int(index), sample_id=str(identifiers[position]),
                                    label=float(labels[position]),
                                    **{"evaluator_{}_pred".format(mode): float(predictions[mode][position])
                                       for mode in ("LAV",) + MISSING_MODES}))
    if not records:
        raise RuntimeError("The train cache loader was empty.")
    frame = pd.DataFrame(records).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if frame.sample_index.duplicated().any() or not np.array_equal(frame.sample_index.to_numpy(), np.arange(len(frame))):
        raise RuntimeError("Every train sample must occur once with a unique index.")
    for mode in MISSING_MODES:
        delta = np.abs(frame.evaluator_LAV_pred.to_numpy() - frame["evaluator_{}_pred".format(mode)].to_numpy())
        rank, q, compat = compatibility_from_deltas(delta)
        frame["delta_{}".format(mode)], frame["rank_{}".format(mode)] = delta, rank
        frame["q_{}".format(mode)], frame["compat_{}".format(mode)] = q, compat
    return frame.loc[:, CACHE_COLUMNS]


def write_counterfactual_cache(frame, root, dataset, checkpoint, best_epoch):
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Cache columns do not match the pre-registered schema.")
    paths = cache_paths(root, dataset)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    config = {
        "method": "Stage 3B Counterfactual Compatibility-Gated Prediction KD",
        "version": CACHE_VERSION, "dataset": dataset, "train_sample_count": int(len(frame)),
        "rank_method": "average", "sort_kind": "mergesort", "transform": RANK_TRANSFORM,
        "compatibility": "1-q", "source": "train_only", "evaluator_checkpoint": str(checkpoint),
        "evaluator_sha256": checkpoint_sha256(checkpoint), "evaluator_size_bytes": int(Path(checkpoint).stat().st_size),
        "evaluator_key_count": int(len(torch.load(checkpoint, map_location="cpu"))),
        "evaluator_best_epoch": int(best_epoch),
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["config_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    frame.to_csv(paths["csv"], index=False)
    paths["config"].write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    paths["summary"].write_text(json.dumps(cache_summary(frame), indent=2, sort_keys=True) + "\n")
    pd.DataFrame(cache_bin_rows(frame)).to_csv(paths["bins"], index=False)
    return paths, config


def load_counterfactual_cache(root, dataset):
    paths = cache_paths(root, dataset)
    if not paths["csv"].is_file() or not paths["config"].is_file():
        raise FileNotFoundError("Counterfactual cache has not been built.")
    frame = pd.read_csv(paths["csv"])
    if list(frame.columns) != list(CACHE_COLUMNS) or frame.sample_index.duplicated().any():
        raise ValueError("Counterfactual cache is malformed.")
    for mode in MISSING_MODES:
        values = frame["compat_{}".format(mode)].to_numpy()
        if not np.all((values > 0) & (values < 1)):
            raise ValueError("Cached compatibility is outside (0,1).")
    return frame, {int(row.sample_index): row._asdict() for row in frame.itertuples(index=False)}
