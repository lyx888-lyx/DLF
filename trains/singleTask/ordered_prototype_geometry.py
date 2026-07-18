"""Frozen geometry utilities for Stage 11 CC-OPT."""
import hashlib
from contextlib import contextmanager

import numpy as np


LEVELS = np.arange(-3, 4, dtype=np.int64)
MODES = ("LAV", "LA", "LV", "L")
MISSING_MODES = ("LA", "LV", "L")


def acc7_bins(values):
    """Reuse the evaluator's float32 clip + NumPy round-to-even mapping."""
    values = np.asarray(values, dtype=np.float32)
    return np.round(np.clip(values, -3.0, 3.0)).astype(np.int64)


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 2:
        return 0.0
    left_rank = _average_ranks(left[finite])
    right_rank = _average_ranks(right[finite])
    if left_rank.std() == 0 or right_rank.std() == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def compute_prototypes(representations, labels):
    representations = np.asarray(representations, dtype=np.float64)
    bins = acc7_bins(labels)
    prototypes = {}
    counts = {}
    variances = {}
    for level in LEVELS:
        selected = representations[bins == level]
        counts[int(level)] = int(len(selected))
        if not len(selected):
            prototypes[int(level)] = None
            variances[int(level)] = None
            continue
        prototype = selected.mean(axis=0)
        prototypes[int(level)] = prototype
        variances[int(level)] = float(
            np.mean(np.sum((selected - prototype) ** 2, axis=1))
        )
    return prototypes, counts, variances


def valid_prototype_matrix(prototypes):
    levels = [int(level) for level in LEVELS if prototypes[int(level)] is not None]
    return np.asarray(levels), np.stack([prototypes[level] for level in levels])


def geometry_summary(representations, labels, prototypes):
    representations = np.asarray(representations, dtype=np.float64)
    bins = acc7_bins(labels)
    levels, matrix = valid_prototype_matrix(prototypes)
    distances = []
    for level in levels:
        selected = representations[bins == level]
        distances.extend(
            np.linalg.norm(selected - prototypes[int(level)], axis=1).tolist()
        )
    intra = float(np.mean(distances))
    pairwise = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    upper = pairwise[np.triu_indices(len(matrix), 1)]
    separation = float(upper.mean()) if len(upper) else 0.0
    return {
        "IntraClassCompactness": intra,
        "InterClassSeparation": separation,
        "CompactnessSeparationRatio": (
            intra / separation if separation > 0 else float("inf")
        ),
    }


def fit_sentiment_axis(lav_prototypes):
    levels, matrix = valid_prototype_matrix(lav_prototypes)
    design = np.column_stack([matrix, np.ones(len(matrix))])
    solution = np.linalg.lstsq(design, levels.astype(np.float64), rcond=None)[0]
    axis = solution[:-1]
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("Train-only sentiment axis is degenerate.")
    return axis / norm


def ordinal_spearman(prototypes, axis):
    levels, matrix = valid_prototype_matrix(prototypes)
    return spearman(matrix @ axis, levels)


def cross_mode_retrieval(missing_prototypes, lav_prototypes):
    lav_levels, lav_matrix = valid_prototype_matrix(lav_prototypes)
    correct, errors, rows = 0, [], []
    for level in LEVELS:
        prototype = missing_prototypes[int(level)]
        if prototype is None:
            continue
        distances = np.linalg.norm(lav_matrix - prototype, axis=1)
        nearest = int(lav_levels[int(np.argmin(distances))])
        error = abs(nearest - int(level))
        correct += int(nearest == int(level))
        errors.append(error)
        rows.append((int(level), nearest, float(error)))
    return {
        "SameBinRetrievalCount": int(correct),
        "ValidBinCount": int(len(rows)),
        "SameBinRetrievalRate": float(correct / len(rows)) if rows else 0.0,
        "OrderedRetrievalError": float(np.mean(errors)) if errors else float("nan"),
        "Rows": rows,
    }


def prototype_temperature(prototypes):
    _, matrix = valid_prototype_matrix(prototypes)
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    values = distances[np.triu_indices(len(matrix), 1)]
    values = values[values > 0]
    if not len(values):
        raise RuntimeError("Prototype temperature cannot be determined.")
    return float(np.median(values))


def prototype_estimate(representations, prototypes, temperature):
    levels, matrix = valid_prototype_matrix(prototypes)
    distances = np.linalg.norm(
        np.asarray(representations, dtype=np.float64)[:, None, :]
        - matrix[None, :, :],
        axis=2,
    )
    logits = -distances / max(float(temperature), np.finfo(np.float64).eps)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ levels.astype(np.float64)


def displacement(representations, labels, prototypes):
    bins = acc7_bins(labels)
    result = np.full(len(bins), np.nan, dtype=np.float64)
    for index, level in enumerate(bins):
        prototype = prototypes[int(level)]
        if prototype is not None:
            result[index] = np.linalg.norm(representations[index] - prototype)
    return result


class FusionRepresentationCapture:
    """Read the shared pre-regression-head fusion tensor via a read-only hook."""

    def __init__(self, wrapped_model):
        backbone = getattr(wrapped_model, "backbone", wrapped_model)
        self.tensor = None
        self.handle = backbone.proj1.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self.tensor = inputs[0]

    def close(self):
        self.handle.remove()

    def pop(self):
        if self.tensor is None:
            raise RuntimeError("Fusion representation hook did not fire.")
        value = self.tensor
        self.tensor = None
        return value


def stage11a_gate(seed_mode_rows, diagnostics):
    import pandas as pd

    geometry = pd.DataFrame(seed_mode_rows)
    diagnostic = pd.DataFrame(diagnostics)
    lav_ordinal = int(
        (
            geometry.loc[geometry.Mode == "LAV", "OrdinalSpearman"] >= 0.80
        ).sum()
    )
    missing_ordinal_seed_count = 0
    for seed, frame in geometry.loc[
        geometry.Mode.isin(MISSING_MODES)
    ].groupby("Seed"):
        missing_ordinal_seed_count += int(
            (frame.OrdinalSpearman >= 0.60).sum() >= 2
        )
    mean_missing = geometry.loc[
        geometry.Mode.isin(MISSING_MODES)
    ].groupby("Mode").mean(numeric_only=True)
    retrieval_modes = int(
        (
            mean_missing.SameBinRetrievalCount
            >= (5.0 / 7.0) * mean_missing.ValidBinCount
        ).sum()
    )
    ordered_modes = int((mean_missing.OrderedRetrievalError <= 0.75).sum())
    association_modes = 0
    for mode in MISSING_MODES:
        frame = geometry.loc[geometry.Mode == mode]
        association_modes += int(
            frame.ErrorAssociationSpearman.mean() >= 0.15
            and (frame.ErrorAssociationSpearman > 0).sum() >= 3
        )
    per_seed = diagnostic.groupby("Seed").first()
    conditions = {
        "LAVOrdinalAtLeast4Seeds": lav_ordinal >= 4,
        "MissingOrdinalAtLeast4Seeds": missing_ordinal_seed_count >= 4,
        "SameBinRetrievalAtLeast2Modes": retrieval_modes >= 2,
        "OrderedRetrievalAtLeast2Modes": ordered_modes >= 2,
        "ErrorAssociationAtLeast2Modes": association_modes >= 2,
        "MeanJValidDeltaAtMostMinus0.0015": per_seed.DeltaJ.mean() <= -0.0015,
        "JValidImprovedAtLeast4Seeds": int((per_seed.DeltaJ < 0).sum()) >= 4,
        "MeanLAVMAENonDegraded": per_seed.DeltaLAVMAE.mean() <= 0,
        "MeanMissingMacroMAENonDegraded": (
            per_seed.DeltaMissingMacroMAE.mean() <= 0
        ),
        "MeanAcc7DropWithin0.003": per_seed.DeltaMeanAcc7.mean() >= -0.003,
        "MeanAcc5DropWithin0.003": per_seed.DeltaMeanAcc5.mean() >= -0.003,
    }
    conditions = {name: bool(value) for name, value in conditions.items()}
    return {
        "Passed": bool(all(conditions.values())),
        "Verdict": (
            "STAGE11A_POSITIVE_GEOMETRY_SIGNAL"
            if all(conditions.values())
            else "STAGE11A_NO_POSITIVE_GEOMETRY_SIGNAL"
        ),
        "Conditions": conditions,
        "Counts": {
            "LAVOrdinalSeeds": lav_ordinal,
            "MissingOrdinalSeeds": missing_ordinal_seed_count,
            "RetrievalModes": retrieval_modes,
            "OrderedErrorModes": ordered_modes,
            "ErrorAssociationModes": association_modes,
            "ImprovedJSeeds": int((per_seed.DeltaJ < 0).sum()),
        },
        "Means": {
            "DeltaJ": float(per_seed.DeltaJ.mean()),
            "DeltaLAVMAE": float(per_seed.DeltaLAVMAE.mean()),
            "DeltaMissingMacroMAE": float(
                per_seed.DeltaMissingMacroMAE.mean()
            ),
            "DeltaMeanAcc7": float(per_seed.DeltaMeanAcc7.mean()),
            "DeltaMeanAcc5": float(per_seed.DeltaMeanAcc5.mean()),
        },
    }
