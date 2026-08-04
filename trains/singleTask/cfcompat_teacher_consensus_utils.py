"""Locked utilities for a multi-teacher CFCompatKD Valid-only screen."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .cf_compat_kd_utils import gated_kd_loss
from .fixed_kd_utils import checkpoint_sha256
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_teacher_consensus_valid_screen_v1"
METHOD = "DLF-CFCompatKD-TeacherConsensus-v1"
OUTPUT_TAG = "cfcompat_teacher_consensus_v1"
FORMAL_SEEDS = (1111, 1114)
TEACHER_SEEDS = (1111, 1112, 1113, 1114, 1115)
BASELINE_RUN = "single_teacher_replay"
MEAN_RUN = "ensemble_mean"
CONSENSUS_RUN = "ensemble_consensus"
RUNS = (BASELINE_RUN, MEAN_RUN, CONSENSUS_RUN)
CANDIDATE_RUNS = (MEAN_RUN, CONSENSUS_RUN)
REPLAY_TOLERANCE = 1e-4
MEAN_GAIN_REQUIRED = 0.005
PER_SEED_MAX_DEGRADATION = 0.002
SUPPORTING_GAIN = 0.003
SUPPORTING_EPOCHS = 2
CACHE_COLUMNS = (
    "sample_index",
    "sample_id",
    "label",
    "teacher_1111",
    "teacher_1112",
    "teacher_1113",
    "teacher_1114",
    "teacher_1115",
    "teacher_mean",
    "teacher_variance",
    "consensus_weight",
)


def consensus_cache_paths(root: Path) -> Dict[str, Path]:
    directory = Path(root) / "teacher_consensus_train_cache"
    return {
        "directory": directory,
        "csv": directory / "mosi_train_teacher_consensus.csv",
        "config": directory / "mosi_train_teacher_consensus_config.json",
    }


def consensus_statistics(predictions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(TEACHER_SEEDS):
        raise ValueError(
            "Teacher predictions must have shape [5, N] in the frozen seed order."
        )
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Teacher predictions must be finite and non-empty.")
    mean = values.mean(axis=0)
    variance = np.mean(np.square(values - mean[None, :]), axis=0)
    variance_median = float(np.median(variance))
    if not math.isfinite(variance_median) or variance_median <= 0.0:
        raise RuntimeError(
            "The fixed consensus formula requires a positive train median variance."
        )
    weight = 1.0 / (1.0 + variance / variance_median)
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or not np.isfinite(weight).all()
        or np.any(variance < 0.0)
        or np.any(weight <= 0.0)
        or np.any(weight > 1.0)
    ):
        raise FloatingPointError("Invalid teacher-consensus statistics.")
    return mean, variance, variance_median, weight


def make_consensus_cache(
    metadata: pd.DataFrame,
    predictions_by_seed: Mapping[int, Sequence[float]],
) -> Tuple[pd.DataFrame, float]:
    required = {"sample_index", "sample_id", "label"}
    if not required.issubset(metadata.columns):
        raise ValueError("Consensus metadata lacks required train-sample columns.")
    local = metadata.loc[:, ["sample_index", "sample_id", "label"]].copy()
    local = local.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if local.empty or local.sample_index.duplicated().any():
        raise ValueError("Consensus metadata must contain unique train indices.")
    expected_indices = np.arange(len(local), dtype=np.int64)
    if not np.array_equal(local.sample_index.to_numpy(dtype=np.int64), expected_indices):
        raise ValueError("Consensus train indices must be contiguous from zero.")
    matrix = []
    for seed in TEACHER_SEEDS:
        if int(seed) not in predictions_by_seed:
            raise KeyError("Missing teacher prediction seed {}.".format(seed))
        values = np.asarray(predictions_by_seed[int(seed)], dtype=np.float64).reshape(-1)
        if len(values) != len(local) or not np.isfinite(values).all():
            raise ValueError("Teacher seed {} prediction binding is invalid.".format(seed))
        local["teacher_{}".format(seed)] = values
        matrix.append(values)
    mean, variance, variance_median, weight = consensus_statistics(np.stack(matrix, axis=0))
    local["teacher_mean"] = mean
    local["teacher_variance"] = variance
    local["consensus_weight"] = weight
    return local.loc[:, CACHE_COLUMNS], variance_median


def write_consensus_cache(
    frame: pd.DataFrame,
    paths: Mapping[str, Path],
    variance_median: float,
    teacher_records: Sequence[Mapping[str, object]],
    rng_state_preserved: bool,
) -> Dict[str, object]:
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Teacher-consensus cache schema differs from the frozen schema.")
    if [int(item["seed"]) for item in teacher_records] != list(TEACHER_SEEDS):
        raise ValueError("Teacher checkpoint records are not in the frozen seed order.")
    directory = Path(paths["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["csv"], index=False)
    config = {
        "version": VERSION,
        "method": METHOD,
        "dataset": "mosi",
        "source": "official_train_only",
        "sample_count": int(len(frame)),
        "teacher_seeds": list(TEACHER_SEEDS),
        "teacher_checkpoints": list(teacher_records),
        "teacher_target": "arithmetic_mean_of_five_clean_LAV_predictions",
        "teacher_variance": "population_mean_squared_deviation_across_five_teachers",
        "variance_scale": "median_train_teacher_variance",
        "variance_median": float(variance_median),
        "consensus_weight_formula": "1/(1+variance/median_train_variance)",
        "cache_sha256": checkpoint_sha256(paths["csv"]),
        "rng_state_preserved": bool(rng_state_preserved),
        "official_test_constructed": False,
    }
    Path(paths["config"]).write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config


def load_consensus_cache(
    paths: Mapping[str, Path],
) -> Tuple[pd.DataFrame, Dict[int, Mapping[str, object]], Dict[str, object]]:
    if not Path(paths["csv"]).is_file() or not Path(paths["config"]).is_file():
        raise FileNotFoundError("Teacher-consensus train cache is absent.")
    frame = pd.read_csv(paths["csv"])
    config = json.loads(Path(paths["config"]).read_text(encoding="utf-8"))
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Teacher-consensus cache columns are malformed.")
    if (
        config.get("version") != VERSION
        or config.get("method") != METHOD
        or config.get("source") != "official_train_only"
        or tuple(config.get("teacher_seeds", ())) != TEACHER_SEEDS
        or config.get("official_test_constructed") is not False
        or config.get("rng_state_preserved") is not True
        or int(config.get("sample_count", -1)) != len(frame)
        or config.get("cache_sha256") != checkpoint_sha256(paths["csv"])
    ):
        raise ValueError("Teacher-consensus cache configuration binding is invalid.")
    if len(frame) != 1284 or frame.sample_index.duplicated().any():
        raise ValueError("MOSI consensus cache must contain 1284 unique train samples.")
    if not np.array_equal(
        frame.sample_index.to_numpy(dtype=np.int64), np.arange(len(frame), dtype=np.int64)
    ):
        raise ValueError("Teacher-consensus cache indices are not contiguous.")
    matrix = np.stack(
        [frame["teacher_{}".format(seed)].to_numpy(dtype=np.float64) for seed in TEACHER_SEEDS],
        axis=0,
    )
    mean, variance, variance_median, weight = consensus_statistics(matrix)
    if not math.isclose(
        variance_median, float(config["variance_median"]), rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError("Teacher-consensus median variance does not recompute.")
    for expected, column in (
        (mean, "teacher_mean"),
        (variance, "teacher_variance"),
        (weight, "consensus_weight"),
    ):
        if not np.allclose(
            expected,
            frame[column].to_numpy(dtype=np.float64),
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError("Teacher-consensus column {} does not recompute.".format(column))
    for record in config["teacher_checkpoints"]:
        checkpoint = Path(str(record["path"]))
        if not checkpoint.is_file() or checkpoint_sha256(checkpoint) != record["sha256"]:
            raise FileNotFoundError(
                "Teacher checkpoint binding failed for seed {}.".format(record["seed"])
            )
    by_index = {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }
    return frame, by_index, config


def consensus_for_indices(
    cache_by_index: Mapping[int, Mapping[str, object]],
    indices: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    targets, variances, weights = [], [], []
    for index in indices:
        if int(index) not in cache_by_index:
            raise KeyError("Consensus cache has no train sample index {}.".format(index))
        row = cache_by_index[int(index)]
        targets.append(float(row["teacher_mean"]))
        variances.append(float(row["teacher_variance"]))
        weights.append(float(row["consensus_weight"]))
    target = torch.as_tensor(targets, device=device, dtype=dtype)
    variance = torch.as_tensor(variances, device=device, dtype=dtype)
    weight = torch.as_tensor(weights, device=device, dtype=dtype)
    if (
        not torch.isfinite(target).all()
        or not torch.isfinite(variance).all()
        or not torch.isfinite(weight).all()
        or torch.any(variance < 0)
        or torch.any(weight <= 0)
        or torch.any(weight > 1)
    ):
        raise FloatingPointError("Consensus cache lookup returned invalid values.")
    return target, variance, weight


def method_gate(
    run: str,
    compatibility: torch.Tensor,
    consensus_weight: torch.Tensor,
) -> torch.Tensor:
    compatibility = compatibility.detach().view(-1)
    consensus_weight = consensus_weight.detach().view(-1).to(compatibility)
    if run == MEAN_RUN:
        gate = compatibility
    elif run == CONSENSUS_RUN:
        gate = compatibility * consensus_weight
    else:
        raise ValueError("Consensus gate is defined only for ensemble candidates.")
    if not torch.isfinite(gate).all() or torch.any(gate <= 0) or torch.any(gate > 1):
        raise FloatingPointError("Candidate distillation gate is outside (0,1].")
    return gate.detach()


def ensemble_kd_loss(
    run: str,
    student_prediction: torch.Tensor,
    teacher_mean: torch.Tensor,
    compatibility: torch.Tensor,
    consensus_weight: torch.Tensor,
):
    gate = method_gate(run, compatibility, consensus_weight)
    loss, each = gated_kd_loss(student_prediction, teacher_mean, gate)
    return loss, each, gate


def missing_macro(row: Mapping[str, object], prefix: str = "valid") -> float:
    return float(
        np.mean([float(row["{}_{}_MAE".format(prefix, mode)]) for mode in MISSING_MODES])
    )


def replay_gate(replay: Mapping[str, object], reference: Mapping[str, object]) -> Dict[str, object]:
    differences = {
        "J_valid": abs(float(replay["J_valid"]) - float(reference["J_valid"]))
    }
    for mode in ("LAV",) + MISSING_MODES:
        key = "valid_{}_MAE".format(mode)
        differences[key] = abs(float(replay[key]) - float(reference[key]))
    epoch_match = int(replay["BestValidEpoch"]) == int(reference["BestValidEpoch"])
    return {
        "passed": bool(
            epoch_match
            and all(value <= REPLAY_TOLERANCE for value in differences.values())
        ),
        "epoch_match": bool(epoch_match),
        "tolerance": REPLAY_TOLERANCE,
        "differences": differences,
    }


def per_seed_valid_evidence(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
    epoch_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    gain_j = float(baseline["J_valid"]) - float(candidate["J_valid"])
    gain_lav = float(baseline["valid_LAV_MAE"]) - float(candidate["valid_LAV_MAE"])
    gain_missing = missing_macro(baseline) - missing_macro(candidate)
    mode_degradations = {
        mode: float(candidate["valid_{}_MAE".format(mode)])
        - float(baseline["valid_{}_MAE".format(mode)])
        for mode in ("LAV",) + MISSING_MODES
    }
    supporting = int(
        sum(
            float(row["J_valid"])
            <= float(baseline["J_valid"]) - SUPPORTING_GAIN
            for row in epoch_rows
        )
    )
    return {
        "gain_valid_J": gain_j,
        "gain_valid_LAV_MAE": gain_lav,
        "gain_valid_MissingMacro_MAE": gain_missing,
        "mode_degradations": mode_degradations,
        "supporting_epoch_count": supporting,
        "positive_J_gain": bool(gain_j > 0.0),
        "no_mode_degradation_over_0p002": bool(
            max(mode_degradations.values()) <= PER_SEED_MAX_DEGRADATION
        ),
        "LAV_not_materially_degraded": bool(
            gain_lav >= -PER_SEED_MAX_DEGRADATION
        ),
        "MissingMacro_not_materially_degraded": bool(
            gain_missing >= -PER_SEED_MAX_DEGRADATION
        ),
        "at_least_two_supporting_epochs": bool(supporting >= SUPPORTING_EPOCHS),
    }


def aggregate_candidate_gate(
    run: str,
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Mapping[int, Mapping[str, object]],
    epoch_rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if run not in CANDIDATE_RUNS:
        raise ValueError("Unknown candidate run {}.".format(run))
    selected = [row for row in candidate_rows if str(row["Run"]) == run]
    if sorted(int(row["Seed"]) for row in selected) != sorted(FORMAL_SEEDS):
        raise RuntimeError("Candidate {} lacks the two formal seeds.".format(run))
    evidence, gains = {}, []
    for row in selected:
        seed = int(row["Seed"])
        local_epochs = [
            epoch
            for epoch in epoch_rows
            if int(epoch["Seed"]) == seed and str(epoch["Run"]) == run
        ]
        local = per_seed_valid_evidence(row, baseline_rows[seed], local_epochs)
        evidence[str(seed)] = local
        gains.append(local["gain_valid_J"])
    checks = {
        "both_seeds_positive_J_gain": all(
            item["positive_J_gain"] for item in evidence.values()
        ),
        "mean_J_gain_ge_0p005": float(np.mean(gains)) >= MEAN_GAIN_REQUIRED,
        "no_seed_or_mode_degradation_over_0p002": all(
            item["no_mode_degradation_over_0p002"] for item in evidence.values()
        ),
        "LAV_not_materially_degraded_both_seeds": all(
            item["LAV_not_materially_degraded"] for item in evidence.values()
        ),
        "MissingMacro_not_materially_degraded_both_seeds": all(
            item["MissingMacro_not_materially_degraded"] for item in evidence.values()
        ),
        "two_supporting_epochs_each_seed": all(
            item["at_least_two_supporting_epochs"] for item in evidence.values()
        ),
    }
    return {
        "run": run,
        "passed": bool(all(checks.values())),
        "mean_gain_valid_J": float(np.mean(gains)),
        "checks": checks,
        "per_seed": evidence,
        "required_mean_gain_valid_J": MEAN_GAIN_REQUIRED,
        "max_per_seed_mode_degradation": PER_SEED_MAX_DEGRADATION,
        "required_supporting_epochs_each_seed": SUPPORTING_EPOCHS,
        "official_test_authorized": False,
    }


def verdict_from_gates(gates: Mapping[str, Mapping[str, object]]) -> str:
    if set(gates) != set(CANDIDATE_RUNS):
        raise RuntimeError("Both candidate gates are required for the verdict.")
    if bool(gates[CONSENSUS_RUN]["passed"]):
        return "PROMOTE_TEACHER_CONSENSUS_TO_MOSEI_SINGLE_SEED_VALID_SCREEN"
    if bool(gates[MEAN_RUN]["passed"]):
        return "PROMOTE_ENSEMBLE_MEAN_ABLATION_TO_MOSEI_SINGLE_SEED_VALID_SCREEN"
    return "STOP_TEACHER_CONSENSUS_DUAL_SEED_VALID_FAILED"
