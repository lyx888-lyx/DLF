"""Utilities for a locked two-seed SAM validation screen on CFCompatKD."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .cf_compat_kd_utils import (
    CACHE_COLUMNS,
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    cache_paths,
)
from .missing_utils import MISSING_MODES


VERSION = "cfcompat_sam_valid_screen_v1"
METHOD = "DLF-CFCompatKD-SAM-v1"
OUTPUT_TAG = "cfcompat_sam_v1"
FORMAL_SEEDS = (1111, 1114)
SAM_RHOS = (0.01, 0.05)
REPLAY_TOLERANCE = 1e-4
MEAN_GAIN_REQUIRED = 0.005
PER_SEED_MAX_DEGRADATION = 0.002
SUPPORTING_GAIN = 0.003
SUPPORTING_EPOCHS = 2


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_states_equal(left: Mapping, right: Mapping) -> bool:
    if left["python"] != right["python"]:
        return False
    if left["numpy"][0] != right["numpy"][0]:
        return False
    if not np.array_equal(left["numpy"][1], right["numpy"][1]):
        return False
    if left["numpy"][2:] != right["numpy"][2:]:
        return False
    if not torch.equal(left["torch"], right["torch"]):
        return False
    return len(left["cuda"]) == len(right["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"])
    )


class SAMController:
    """Parameter perturbation controller for SAM with an external optimizer.

    First-pass gradients are accumulated using the original CFCompatKD
    ``update_epochs`` window. ``ascent_step`` perturbs trainable parameters.
    After the second pass, ``descent_step`` restores the original parameters and
    applies exactly one base-optimizer update using second-pass gradients.
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], rho: float):
        self.parameters = [
            parameter for parameter in parameters if parameter.requires_grad
        ]
        self.rho = float(rho)
        if self.rho <= 0:
            raise ValueError("SAM rho must be positive.")
        if not self.parameters:
            raise ValueError("SAM received no trainable parameters.")
        self._perturbations: Dict[int, torch.Tensor] = {}

    def grad_norm(self) -> torch.Tensor:
        norms = [
            parameter.grad.detach().norm(p=2)
            for parameter in self.parameters
            if parameter.grad is not None
        ]
        if not norms:
            return torch.zeros((), device=self.parameters[0].device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def ascent_step(self) -> float:
        norm = self.grad_norm()
        if not torch.isfinite(norm) or float(norm) <= 0:
            raise RuntimeError(
                "SAM first-pass gradient norm is non-positive or non-finite."
            )
        scale = self.rho / (norm + 1e-12)
        self._perturbations.clear()
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            perturbation = parameter.grad.detach() * scale.to(parameter)
            if not torch.isfinite(perturbation).all():
                raise FloatingPointError("SAM perturbation is non-finite.")
            parameter.add_(perturbation)
            self._perturbations[id(parameter)] = perturbation
        return float(norm.detach().cpu())

    @torch.no_grad()
    def restore(self) -> None:
        if not self._perturbations:
            raise RuntimeError("SAM restore called before ascent_step.")
        for parameter in self.parameters:
            perturbation = self._perturbations.get(id(parameter))
            if perturbation is not None:
                parameter.sub_(perturbation)
        self._perturbations.clear()

    @torch.no_grad()
    def descent_step(self, optimizer: torch.optim.Optimizer) -> None:
        self.restore()
        optimizer.step()


def portable_locate_stage1_evaluator(
    result_root,
    dataset,
    seed,
    multiseed: bool = False,
    smoke: bool = False,
):
    """Resolve historical Stage-1 checkpoint paths against the asset project."""
    if multiseed:
        source = (
            Path(result_root)
            / "missing_baseline"
            / "moddrop_benchmark_multiseed_v1"
        )
        if smoke:
            source = source / "smoke"
        source = source / f"seed{int(seed)}" / f"{dataset}_per_seed.csv"
        checkpoint_field, epoch_field = "MainCheckpoint", "BestValidEpoch"
    else:
        source = (
            Path(result_root)
            / "missing_baseline"
            / "moddrop"
            / "train"
            / f"{dataset}_per_seed.csv"
        )
        checkpoint_field, epoch_field = "Checkpoint", "BestEpoch"
    if not source.is_file():
        raise FileNotFoundError(f"Required Stage-1 result CSV absent: {source}")
    rows = pd.read_csv(source)
    selected = rows.loc[rows.Seed.astype(int).eq(int(seed))]
    if len(selected) != 1 or checkpoint_field not in selected:
        raise ValueError(
            f"Stage-1 CSV has no unique checkpoint for seed {seed}."
        )
    checkpoint = Path(str(selected.iloc[0][checkpoint_field]))
    if not checkpoint.is_absolute() and not checkpoint.is_file():
        rooted = Path(result_root).resolve().parent / checkpoint
        if rooted.is_file():
            checkpoint = rooted
    if multiseed and (
        "diagnostic" in str(checkpoint) or "best_test" in str(checkpoint)
    ):
        raise ValueError(
            "Counterfactual evaluator must be validation-best ModDrop."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-1 checkpoint is absent: {checkpoint}")
    return checkpoint, int(selected.iloc[0][epoch_field]), source


def load_locked_counterfactual_cache(
    root,
    dataset,
    version=CACHE_VERSION,
    seed=None,
    expected_evaluator_sha=None,
):
    """Load current caches and the locked historical seed-1111 format."""
    paths = cache_paths(root, dataset, version=version, seed=seed)
    if not paths["csv"].is_file() or not paths["config"].is_file():
        raise FileNotFoundError("Counterfactual cache has not been built.")
    frame = pd.read_csv(paths["csv"])
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    expected_seed = None if seed is None else int(seed)
    if (
        config.get("version") != version
        or config.get("seed") != expected_seed
    ):
        raise ValueError(
            "Counterfactual cache version/seed binding is invalid."
        )
    if (
        expected_evaluator_sha is not None
        and config.get("evaluator_sha256") != expected_evaluator_sha
    ):
        raise ValueError(
            "Counterfactual cache evaluator SHA binding is invalid."
        )
    if config.get("source") != "train_only":
        raise ValueError("Counterfactual cache is not train-only.")
    legacy_seed1111 = (
        version == CACHE_VERSION
        and expected_seed is None
        and "created_from_train_only" not in config
    )
    if (
        config.get("created_from_train_only") is not True
        and not legacy_seed1111
    ):
        raise ValueError("Counterfactual cache is not train-only.")
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Counterfactual cache schema is malformed.")
    if frame.sample_index.duplicated().any():
        raise ValueError(
            "Counterfactual cache sample indices are duplicated."
        )
    for mode in MISSING_MODES:
        values = frame[f"compat_{mode}"].to_numpy(dtype=float)
        if (
            not np.isfinite(values).all()
            or not np.all((values > 0) & (values < 1))
        ):
            raise ValueError("Cached compatibility is outside (0,1).")
    by_index = {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }
    return frame, by_index, paths


def missing_macro(row: Mapping, prefix: str = "valid") -> float:
    return float(
        np.mean(
            [
                float(row[f"{prefix}_{mode}_MAE"])
                for mode in MISSING_MODES
            ]
        )
    )


def replay_gate(replay: Mapping, reference: Mapping) -> dict:
    differences = {
        "J_valid": abs(
            float(replay["J_valid"]) - float(reference["J_valid"])
        ),
    }
    for mode in ("LAV",) + MISSING_MODES:
        key = f"valid_{mode}_MAE"
        differences[key] = abs(
            float(replay[key]) - float(reference[key])
        )
    epoch_match = int(replay["BestValidEpoch"]) == int(
        reference["BestValidEpoch"]
    )
    return {
        "passed": bool(
            epoch_match
            and all(
                value <= REPLAY_TOLERANCE
                for value in differences.values()
            )
        ),
        "epoch_match": bool(epoch_match),
        "tolerance": REPLAY_TOLERANCE,
        "differences": differences,
    }


def per_seed_valid_evidence(
    candidate: Mapping,
    baseline: Mapping,
    epoch_rows: Sequence[Mapping],
) -> dict:
    gain_j = float(baseline["J_valid"]) - float(candidate["J_valid"])
    gain_lav = float(baseline["valid_LAV_MAE"]) - float(
        candidate["valid_LAV_MAE"]
    )
    gain_missing = missing_macro(baseline) - missing_macro(candidate)
    mode_degradations = {
        mode: float(candidate[f"valid_{mode}_MAE"])
        - float(baseline[f"valid_{mode}_MAE"])
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
        "at_least_two_supporting_epochs": bool(
            supporting >= SUPPORTING_EPOCHS
        ),
    }


def select_rho(rows: Sequence[Mapping]) -> float:
    by_rho: Dict[float, List[Mapping]] = {}
    for row in rows:
        by_rho.setdefault(float(row["Rho"]), []).append(row)
    if set(by_rho) != set(SAM_RHOS):
        raise RuntimeError("SAM rho grid is incomplete.")
    for rho, local in by_rho.items():
        if sorted(int(row["Seed"]) for row in local) != sorted(
            FORMAL_SEEDS
        ):
            raise RuntimeError(
                f"rho={rho} does not contain both formal seeds."
            )
    return min(
        SAM_RHOS,
        key=lambda rho: (
            float(np.mean([row["J_valid"] for row in by_rho[rho]])),
            float(rho),
        ),
    )


def aggregate_valid_gate(
    selected_rho: float,
    candidate_rows: Sequence[Mapping],
    baseline_rows: Mapping[int, Mapping],
    epoch_rows: Sequence[Mapping],
) -> dict:
    selected = [
        row
        for row in candidate_rows
        if math.isclose(
            float(row["Rho"]), float(selected_rho), abs_tol=0.0
        )
    ]
    evidence = {}
    gains = []
    for row in selected:
        seed = int(row["Seed"])
        local_epochs = [
            epoch
            for epoch in epoch_rows
            if int(epoch["Seed"]) == seed
            and math.isclose(
                float(epoch["Rho"]), float(selected_rho), abs_tol=0.0
            )
        ]
        local = per_seed_valid_evidence(
            row, baseline_rows[seed], local_epochs
        )
        evidence[str(seed)] = local
        gains.append(local["gain_valid_J"])
    checks = {
        "both_seeds_positive_J_gain": all(
            item["positive_J_gain"] for item in evidence.values()
        ),
        "mean_J_gain_ge_0p005": float(np.mean(gains))
        >= MEAN_GAIN_REQUIRED,
        "no_seed_or_mode_degradation_over_0p002": all(
            item["no_mode_degradation_over_0p002"]
            for item in evidence.values()
        ),
        "LAV_not_materially_degraded_both_seeds": all(
            item["LAV_not_materially_degraded"]
            for item in evidence.values()
        ),
        "MissingMacro_not_materially_degraded_both_seeds": all(
            item["MissingMacro_not_materially_degraded"]
            for item in evidence.values()
        ),
        "two_supporting_epochs_each_seed": all(
            item["at_least_two_supporting_epochs"]
            for item in evidence.values()
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "selected_rho": float(selected_rho),
        "mean_gain_valid_J": float(np.mean(gains)),
        "checks": checks,
        "per_seed": evidence,
        "required_mean_gain_valid_J": MEAN_GAIN_REQUIRED,
        "max_per_seed_mode_degradation": PER_SEED_MAX_DEGRADATION,
        "required_supporting_epochs_each_seed": SUPPORTING_EPOCHS,
        "test_authorized": False,
        "next_required_stage": (
            "train_only_group_stability_screen"
            if all(checks.values())
            else "stop_sam"
        ),
    }
