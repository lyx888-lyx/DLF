"""Frozen low-capacity CS-DFMRC calibration table implementation."""

import hashlib
import json

import numpy as np

from trains.singleTask.cross_seed_median_residual import sign_agreement


def _stable_seed_median_table(frame, group_columns, seeds, n_min):
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        medians = []
        counts = []
        supported = []
        for seed in seeds:
            selected = group.loc[group.Seed.eq(seed), "SafeResidual"]
            count = len(selected)
            median = (
                float(np.median(selected.to_numpy(np.float64)))
                if count
                else None
            )
            medians.append(median)
            counts.append(count)
            if count >= n_min:
                supported.append(median)
        agreement, direction = (
            sign_agreement(supported) if supported else (0, 0)
        )
        stable = len(supported) >= 3 and agreement >= 3
        row = {
            column: (
                value.item() if isinstance(value, np.generic) else value
            )
            for column, value in zip(group_columns, keys)
        }
        row.update(
            {
                "SeedMedians": {
                    str(seed): median
                    for seed, median in zip(seeds, medians)
                },
                "SeedCounts": {
                    str(seed): int(count)
                    for seed, count in zip(seeds, counts)
                },
                "SupportSeedCount": len(supported),
                "SignAgreementCount": agreement,
                "AgreedSign": direction,
                "Stable": stable,
                "Residual": (
                    float(np.median(supported)) if stable else None
                ),
            }
        )
        rows.append(row)
    return rows


def build_calibrator(train_frame, training_seeds, n_min):
    seeds = tuple(sorted(int(seed) for seed in training_seeds))
    if len(seeds) not in (4, 5):
        raise ValueError("CS-DFMRC expects four LOSO or five frozen seeds.")
    selected = train_frame.loc[train_frame.Seed.isin(seeds)].copy()
    if set(selected.Seed.unique()) != set(seeds):
        raise ValueError("Training seeds are incomplete.")
    local = _stable_seed_median_table(
        selected, ["Mode", "Cell", "Half"], seeds, n_min
    )
    pool = _stable_seed_median_table(
        selected, ["Cell", "Half"], seeds, n_min
    )
    return {
        "Method": "Cross-Seed Decision-Feasible Median Residual Calibrator",
        "TrainingSeeds": list(seeds),
        "NMin": int(n_min),
        "HalfBins": 2,
        "ShrinkageConstant": 2,
        "LocalEntries": local,
        "PooledEntries": pool,
        "Inputs": [
            "train_baseline_prediction",
            "train_mode",
            "train_decision_cell",
            "train_half",
            "train_label",
        ],
        "ForbiddenInputs": [
            "valid_label",
            "test_data",
            "hidden_representation",
            "prototype",
            "OT",
            "OOF",
            "ensemble_prediction",
        ],
    }


def calibrator_sha(calibrator):
    payload = json.dumps(
        calibrator, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serialize_calibrator(calibrator, path):
    path.write_text(json.dumps(calibrator, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(entries, columns):
    result = {}
    for row in entries:
        key = tuple(row[column] for column in columns)
        result[key] = row
    return result


def correction_for_group(calibrator, mode, cell, half):
    local = _index(
        calibrator["LocalEntries"], ("Mode", "Cell", "Half")
    ).get((mode, cell, int(half)))
    pool = _index(
        calibrator["PooledEntries"], ("Cell", "Half")
    ).get((cell, int(half)))
    pool_available = bool(pool and pool["Stable"])
    local_available = bool(local and local["Stable"])
    if not pool_available:
        return 0.0, "Zero", 0.0
    if not local_available:
        return float(pool["Residual"]), "Pooled", 0.0
    support = int(local["SupportSeedCount"])
    weight = support / float(support + 2)
    correction = (
        weight * float(local["Residual"])
        + (1.0 - weight) * float(pool["Residual"])
    )
    return correction, "LocalShrinkage", weight
