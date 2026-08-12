"""One-shot aggregate-only MOSEI final Test for frozen FixedBlend-DP57-v1.

This evaluator is intentionally separated from all development code.  It has two
modes:

* --preflight: verify every frozen source, SHA, validation decision, and model
  state can be reconstructed.  Official Test is NOT constructed.
* --execute-final-test: after the same preflight passes, create the official
  Test exactly once, cache its batches once in CPU memory, evaluate the five
  frozen CFCompat members and the frozen v13 expert sequentially, compose the
  frozen Raw5 -> 0.5/0.5 FixedBlend -> DP57 route in memory, and write only
  aggregate metrics/projection summaries.

No sample-level Test prediction, label, id, error, calibration, or projection
row is written.  There is no Test-time model/seed/weight/anchor selection and no
overwrite/rerun option.  A one-shot access ledger is created immediately before
Official Test construction; if execution fails after that point, the ledger
remains and the evaluator refuses another final-Test attempt.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch

from train_mosei_cfcompat_v1 import build_config as build_mosei_config
from trains.singleTask.anchor_decision_projection import evaluator_decisions, project_array
from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import (
    CONSENSUS_MIN_AGREE,
    N_FOLDS,
    FrozenS0CrossfitConsensus,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    build_single_split_loader,
    mode_to_mask,
    regression_metrics,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


DATASET = "mosei"
SEEDS = (1111, 1112, 1113, 1114, 1115)
MODES = ("LAV",) + MISSING_MODES
V13_SEED = 1113
FROZEN_ANCHOR_SEED = 1114
RAW5_MEMBER_WEIGHT = np.float32(0.2)
BLEND_WEIGHT_RAW5 = np.float32(0.5)
BLEND_WEIGHT_V13 = np.float32(0.5)
DP_VARIANT = "adpep57"
VERSION = "mosei_fixedblend_dp57_v1_final_test"
PRIMARY_METHOD = "FixedBlend-DP57-v1"

# Already-frozen Valid audit values.  These are integrity bindings only; no
# Test value is compared against them for any decision.
FROZEN_VALID_J = {
    "Anchor": 0.523081084,
    "Raw5": 0.513192346,
    "v13": 0.531753172,
    "FixedBlend": 0.515409470,
    "DP57": 0.515621195,
}
VALID_J_TOL = 5e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot aggregate-only MOSEI FixedBlend-DP57-v1 final Test"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute-final-test", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="high"
    )
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    args = parser.parse_args()
    if int(args.num_workers) != 0:
        parser.error("Windows final Test fixes --num-workers=0.")
    # build_mosei_config expects a seed even though every final source is fixed.
    args.seed = V13_SEED
    args.gpu_ids = [int(args.gpu_id)]
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def final_output_dir(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / "fixedblend_dp57_v1"
        / DATASET
        / "final_test"
    )


def cfcompat_checkpoint(cli: argparse.Namespace, seed: int) -> Path:
    return (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cf_compat_kd_v1"
        / DATASET
        / f"seed{seed}"
        / f"DLF_{DATASET}_seed{seed}_best_valid.pth"
    )


def v13_root(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_adam_step_safety_v13"
        / DATASET
        / "valid_screen"
        / "seed1113_dev"
    )


def v13_checkpoint(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cfcompat_adam_step_safety_v13"
        / DATASET
        / "valid_screen"
        / "seed1113_dev"
        / "frozen_consensus_valid_ready.pth"
    )


def valid_composition_summary(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / "fixedblend_dp57_v1"
        / DATASET
        / "valid_screen"
        / "valid_summary.json"
    )


def canonical_cfcompat_results(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / "cf_compat_kd_v1"
        / f"{DATASET}_per_seed.csv"
    )


def _close(a: float, b: float, tol: float = VALID_J_TOL) -> bool:
    return bool(abs(float(a) - float(b)) <= float(tol))


def verify_frozen_valid_protocol(cli: argparse.Namespace) -> Dict[str, Any]:
    path = valid_composition_summary(cli)
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen Valid composition summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    protocol = summary.get("protocol", {})

    if str(summary.get("dataset", "")).lower() != DATASET:
        raise RuntimeError("Frozen Valid composition dataset binding failed.")
    if str(summary.get("method", "")) != PRIMARY_METHOD:
        raise RuntimeError("Frozen Valid primary-method binding failed.")
    if tuple(int(x) for x in protocol.get("raw5_members", [])) != SEEDS:
        raise RuntimeError("Frozen Raw5 membership drifted.")
    if not _close(protocol.get("raw5_equal_weight", np.nan), RAW5_MEMBER_WEIGHT, 1e-9):
        raise RuntimeError("Frozen Raw5 member weight drifted.")
    if int(protocol.get("v13_seed", -1)) != V13_SEED:
        raise RuntimeError("Frozen v13 seed drifted.")
    weights = protocol.get("blend_weights", {})
    if not _close(weights.get("Raw5", np.nan), BLEND_WEIGHT_RAW5, 1e-9):
        raise RuntimeError("Frozen Raw5 blend weight drifted.")
    if not _close(weights.get("v13", np.nan), BLEND_WEIGHT_V13, 1e-9):
        raise RuntimeError("Frozen v13 blend weight drifted.")
    if bool(protocol.get("blend_weight_search", True)):
        raise RuntimeError("Frozen Valid summary unexpectedly permits blend-weight search.")
    if int(protocol.get("anchor_seed", -1)) != FROZEN_ANCHOR_SEED:
        raise RuntimeError("Frozen DP57 anchor drifted.")
    if str(protocol.get("anchor_selected_by", "")) != "minimum_CFCompat_validation_J":
        raise RuntimeError("Frozen anchor selection rule drifted.")
    if str(protocol.get("dp_variant", "")) != DP_VARIANT:
        raise RuntimeError("Frozen DP variant drifted.")
    if bool(protocol.get("labels_used_by_dp_projection", True)):
        raise RuntimeError("Frozen DP summary unexpectedly uses labels in projection.")
    if bool(protocol.get("official_test_constructed", True)) or bool(
        protocol.get("official_test_accessed", True)
    ):
        raise RuntimeError("Frozen Valid composition did not preserve Test isolation.")

    metrics = summary.get("metrics", {})
    for name, expected in FROZEN_VALID_J.items():
        observed = metrics.get(name, {}).get("J", np.nan)
        if not _close(observed, expected):
            raise RuntimeError(
                f"Frozen Valid J drifted for {name}: observed={observed} expected={expected}"
            )
    return {"path": str(path), "sha256": sha256(path), "summary": summary}


def verify_cfcompat_sources(cli: argparse.Namespace) -> Dict[int, Dict[str, Any]]:
    per_seed_path = canonical_cfcompat_results(cli)
    if not per_seed_path.is_file():
        raise FileNotFoundError(per_seed_path)
    frame = pd.read_csv(per_seed_path)
    required = {"Seed", "J_valid", "CheckpointSHA256", "TestConstructed"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Canonical CFCompat CSV lacks columns: {sorted(missing)}")
    local = frame.loc[frame.Seed.astype(int).isin(SEEDS)].copy()
    if len(local) != len(SEEDS) or local.Seed.astype(int).nunique() != len(SEEDS):
        raise RuntimeError("Canonical CFCompat source rows are incomplete or duplicated.")
    if local.TestConstructed.astype(str).str.lower().isin(("true", "1")).any():
        raise RuntimeError("A formal CFCompat source unexpectedly constructed Test during training.")

    selected = int(
        local.assign(SeedInt=local.Seed.astype(int))
        .sort_values(["J_valid", "SeedInt"], kind="mergesort")
        .iloc[0]
        .SeedInt
    )
    if selected != FROZEN_ANCHOR_SEED:
        raise RuntimeError(
            f"Validation-selected anchor drifted: {selected} != {FROZEN_ANCHOR_SEED}"
        )

    result: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        row = local.loc[local.Seed.astype(int).eq(seed)].iloc[0]
        checkpoint = cfcompat_checkpoint(cli, seed)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_sha = sha256(checkpoint)
        recorded_sha = str(row.CheckpointSHA256).lower()
        if actual_sha != recorded_sha:
            raise RuntimeError(
                f"CFCompat checkpoint SHA mismatch seed{seed}: actual={actual_sha} recorded={recorded_sha}"
            )
        result[seed] = {
            "checkpoint": str(checkpoint),
            "sha256": actual_sha,
            "J_valid": float(row.J_valid),
        }
    return result


def verify_v13_source(cli: argparse.Namespace) -> Dict[str, Any]:
    root = v13_root(cli)
    summary_path = root / "adam_step_safety_v13_mosei_summary.json"
    checkpoint = v13_checkpoint(cli)
    if not summary_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing formal v13 source: {summary_path} / {checkpoint}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("seed", -1)) != V13_SEED or str(summary.get("dataset", "")) != DATASET:
        raise RuntimeError("Formal v13 dataset/seed binding failed.")
    if str(summary.get("verdict", "")).upper().startswith("SMOKE"):
        raise RuntimeError("Refusing smoke v13 checkpoint for final Test.")
    protocol = summary.get("protocol", {})
    if bool(protocol.get("official_test_constructed", True)) or bool(
        protocol.get("official_test_accessed", True)
    ):
        raise RuntimeError("Formal v13 source does not certify Test isolation.")
    if bool(protocol.get("mosei_blend_weight_search_allowed", True)):
        raise RuntimeError("Formal v13 protocol unexpectedly allows MOSEI blend search.")
    observed_j = float(summary.get("candidate_J_valid", np.nan))
    if not _close(observed_j, FROZEN_VALID_J["v13"]):
        raise RuntimeError(f"Formal v13 Valid J drifted: {observed_j}")
    actual_sha = sha256(checkpoint)
    binding = summary.get("frozen_consensus_checkpoint", {})
    recorded_sha = str(binding.get("sha256", "")).lower()
    if actual_sha != recorded_sha:
        raise RuntimeError(
            f"Formal v13 checkpoint SHA mismatch: actual={actual_sha} recorded={recorded_sha}"
        )
    return {
        "checkpoint": str(checkpoint),
        "sha256": actual_sha,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
    }


def build_cfcompat_model(args: Any, checkpoint: Path) -> MissingModalityWrapper:
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _extract_v13_bank_states(state: Dict[str, torch.Tensor]) -> list[Dict[str, torch.Tensor]]:
    banks = []
    for fold in range(N_FOLDS):
        prefix = f"fold_banks.{fold}."
        local = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if str(key).startswith(prefix)
        }
        if not local:
            raise RuntimeError(f"Formal v13 checkpoint has no fold bank {fold}.")
        banks.append(local)
    extra_fold_keys = [
        key
        for key in state
        if str(key).startswith("fold_banks.")
        and int(str(key).split(".")[1]) >= N_FOLDS
    ]
    if extra_fold_keys:
        raise RuntimeError("Formal v13 checkpoint contains unexpected extra fold banks.")
    return banks


def build_v13_model(args: Any, checkpoint: Path) -> FrozenS0CrossfitConsensus:
    state = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state, dict):
        raise RuntimeError("Formal v13 checkpoint is not a state_dict mapping.")
    bank_states = _extract_v13_bank_states(state)
    backbone = DLF(args).to(args.device)
    s0 = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    candidate = FrozenS0CrossfitConsensus(
        s0,
        bank_states,
        min_agree=CONSENSUS_MIN_AGREE,
    ).to(args.device)
    candidate.load_state_dict(state, strict=True)
    candidate.eval()
    if any(parameter.requires_grad for parameter in candidate.parameters()):
        raise RuntimeError("Final frozen v13 model unexpectedly has trainable parameters.")
    return candidate


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_model_reconstruction(
    cli: argparse.Namespace,
    args: Any,
    cf_sources: Dict[int, Dict[str, Any]],
    v13_source: Dict[str, Any],
) -> None:
    # Strict-load every final model before Official Test can be constructed.
    for seed in SEEDS:
        model = build_cfcompat_model(args, Path(cf_sources[seed]["checkpoint"]))
        release_model(model)
    model = build_v13_model(args, Path(v13_source["checkpoint"]))
    release_model(model)


def verify_sources(cli: argparse.Namespace, args: Any, reconstruct_models: bool) -> Dict[str, Any]:
    valid = verify_frozen_valid_protocol(cli)
    cf = verify_cfcompat_sources(cli)
    v13 = verify_v13_source(cli)
    if reconstruct_models:
        verify_model_reconstruction(cli, args, cf, v13)
    return {
        "valid_composition": {"path": valid["path"], "sha256": valid["sha256"]},
        "cfcompat": cf,
        "v13": v13,
    }


def _clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().clone()


def cache_official_test_once(loader) -> tuple[list[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Read the Test DataLoader exactly once and retain only an in-memory batch cache."""
    batches: list[Dict[str, Any]] = []
    labels = []
    indices = []
    observed_ids = set()
    for batch in loader:
        label = _clone_cpu(batch["labels"]["M"].view(-1, 1))
        index = _clone_cpu(batch["index"].view(-1)).to(torch.int64)
        ids = tuple(str(value) for value in batch["id"])
        if len(ids) != int(index.numel()) or int(label.size(0)) != int(index.numel()):
            raise RuntimeError("Final Test batch id/index/label cardinality mismatch.")
        for sample_id in ids:
            if sample_id in observed_ids:
                raise RuntimeError("Duplicate sample_id in Official Test cache.")
            observed_ids.add(sample_id)
        batches.append(
            {
                "text": _clone_cpu(batch["text"]),
                "audio": _clone_cpu(batch["audio"]),
                "vision": _clone_cpu(batch["vision"]),
                "labels": label,
                "index": index,
                # IDs are retained only in RAM for integrity and never written.
                "ids": ids,
            }
        )
        labels.append(label.view(-1).numpy().astype(np.float32, copy=False))
        indices.append(index.numpy().astype(np.int64, copy=False))

    if not batches:
        raise RuntimeError("Official Test loader was empty.")
    label_array = np.concatenate(labels).astype(np.float32, copy=False)
    index_array = np.concatenate(indices).astype(np.int64, copy=False)
    if len(label_array) != len(loader.dataset):
        raise RuntimeError(
            f"Official Test cache length mismatch: cached={len(label_array)} dataset={len(loader.dataset)}"
        )
    if len(np.unique(index_array)) != len(index_array):
        raise RuntimeError("Duplicate sample_index in Official Test cache.")
    if not np.array_equal(np.sort(index_array), np.arange(len(index_array), dtype=np.int64)):
        raise RuntimeError("Official Test sample_index must be a permutation of 0..N-1.")
    if not np.isfinite(label_array).all():
        raise FloatingPointError("Official Test labels contain NaN/Inf.")
    return batches, label_array, index_array


def infer_cached_batches(model, batches: Iterable[Dict[str, Any]], device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, list[np.ndarray]] = {mode: [] for mode in MODES}
    with torch.inference_mode():
        for batch in batches:
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            batch_size = int(text.size(0))
            for mode in MODES:
                mask = mode_to_mask(mode, batch_size, device, audio.dtype)
                prediction = model(text, audio, vision, mask)["output_logit"].view(-1)
                values = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite final Test prediction mode={mode}.")
                collected[mode].append(values.copy())
            del text, audio, vision
    return {
        mode: np.concatenate(collected[mode]).astype(np.float32, copy=False)
        for mode in MODES
    }


def evaluate_prediction_set(predictions: Dict[str, np.ndarray], labels: np.ndarray) -> tuple[Dict[str, Dict[str, float]], float]:
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    metrics: Dict[str, Dict[str, float]] = {}
    for mode in MODES:
        values = np.asarray(predictions[mode], dtype=np.float32).reshape(-1)
        if len(values) != len(labels):
            raise RuntimeError(f"Prediction/label length mismatch mode={mode}.")
        metrics[mode] = regression_metrics(
            torch.as_tensor(values, dtype=torch.float32), label_tensor
        )
    metrics["MissingMacro"] = {
        key: float(np.mean([metrics[mode][key] for mode in MISSING_MODES]))
        for key in metrics["LAV"]
    }
    j = 0.5 * float(metrics["LAV"]["MAE"]) + 0.5 * float(
        metrics["MissingMacro"]["MAE"]
    )
    return metrics, float(j)


def compose_frozen_method(
    member_predictions: Dict[int, Dict[str, np.ndarray]],
    v13_predictions: Dict[str, np.ndarray],
) -> tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, Any]],
]:
    raw5: Dict[str, np.ndarray] = {}
    for mode in MODES:
        stacked = np.stack(
            [np.asarray(member_predictions[seed][mode], dtype=np.float32) for seed in SEEDS],
            axis=0,
        )
        # Equal 0.2 membership.  Accumulate in float32 because direct model
        # predictions are already float32 and no CSV decimal replay is involved.
        raw5[mode] = np.sum(
            stacked * RAW5_MEMBER_WEIGHT, axis=0, dtype=np.float32
        ).astype(np.float32)

    fixedblend = {
        mode: (
            BLEND_WEIGHT_RAW5 * raw5[mode]
            + BLEND_WEIGHT_V13 * np.asarray(v13_predictions[mode], dtype=np.float32)
        ).astype(np.float32)
        for mode in MODES
    }
    anchor = {
        mode: np.asarray(member_predictions[FROZEN_ANCHOR_SEED][mode], dtype=np.float32)
        for mode in MODES
    }

    dp57: Dict[str, np.ndarray] = {}
    projection_summary: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        projected, results = project_array(anchor[mode], fixedblend[mode], DATASET, DP_VARIANT)
        dp57[mode] = projected.astype(np.float32)
        anchor7, anchor5, _ = evaluator_decisions(anchor[mode], DATASET)
        projected7, projected5, _ = evaluator_decisions(dp57[mode], DATASET)
        if not np.array_equal(anchor7, projected7):
            raise RuntimeError(f"Final Test DP57 Acc7 inheritance failed mode={mode}.")
        if not np.array_equal(anchor5, projected5):
            raise RuntimeError(f"Final Test DP57 Acc5 inheritance failed mode={mode}.")
        _, _, blend2 = evaluator_decisions(fixedblend[mode], DATASET)
        _, _, dp2 = evaluator_decisions(dp57[mode], DATASET)
        feasible = np.asarray([result.pe5_already_feasible for result in results], dtype=bool)
        boundary = np.asarray([result.boundary_adjusted for result in results], dtype=bool)
        fallback = np.asarray([result.fallback_to_anchor for result in results], dtype=bool)
        projection_summary[mode] = {
            "N": int(len(results)),
            "projected_fraction": float((~feasible).mean()),
            "target_feasible_fraction": float(feasible.mean()),
            "boundary_adjusted_count": int(boundary.sum()),
            "fallback_count": int(fallback.sum()),
            "acc2_changed_vs_fixedblend_count": int(np.sum(blend2 != dp2)),
        }
    return anchor, raw5, fixedblend, dp57, projection_summary


def metric_line(name: str, metrics: Dict[str, Dict[str, float]], j: float) -> str:
    lav = metrics["LAV"]
    mm = metrics["MissingMacro"]
    return (
        f"{name:12s} J={j:.9f} "
        f"LAV_MAE={lav['MAE']:.9f} MissingMacro_MAE={mm['MAE']:.9f} "
        f"LAV_Corr={lav['Corr']:.9f} LAV_Acc2={lav['acc_2']:.9f} "
        f"LAV_F1={lav['F1_score']:.9f} LAV_Acc7={lav['acc_7']:.9f} "
        f"LAV_Acc5={lav['acc_5']:.9f}"
    )


def aggregate_row(name: str, metrics: Dict[str, Dict[str, float]], j: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "Method": name,
        "PrimaryMethod": bool(name == "DP57"),
        "J": float(j),
    }
    for mode, local in metrics.items():
        for key, value in local.items():
            row[f"{mode}_{key}"] = float(value)
    return row


def run_preflight(cli: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final model reconstruction.")
    setup_seed(V13_SEED)
    args = build_mosei_config(cli)
    sources = verify_sources(cli, args, reconstruct_models=True)
    out = final_output_dir(cli)
    if out.exists():
        raise FileExistsError(
            f"Final-Test output/ledger already exists; one-shot evaluator refuses reuse: {out}"
        )
    print("================ MOSEI FINAL TEST PREFLIGHT =======================")
    print("primary method:", PRIMARY_METHOD)
    print("Raw5 members:", SEEDS, "equal weight=0.2")
    print("v13 seed:", V13_SEED)
    print("blend:", "Raw5 0.5 + v13 0.5")
    print("anchor seed:", FROZEN_ANCHOR_SEED)
    print("DP variant:", DP_VARIANT, "preserves Acc7/Acc5 only")
    print("CFCompat checkpoints verified:", len(sources["cfcompat"]))
    print("v13 checkpoint sha256:", sources["v13"]["sha256"])
    print("Valid frozen summary verified:", sources["valid_composition"]["sha256"])
    print("sample-level Test outputs:", False)
    print("TEST CONSTRUCTED:", False)
    print("STATUS: READY_FOR_ONE_SHOT_FINAL_TEST")


def execute_final_test(cli: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final Test.")
    setup_seed(V13_SEED)
    args = build_mosei_config(cli)

    # Everything that can fail without seeing Test must pass first.
    sources = verify_sources(cli, args, reconstruct_models=True)
    out = final_output_dir(cli)
    if out.exists():
        raise FileExistsError(
            f"Final-Test output/ledger already exists; rerun is forbidden by protocol: {out}"
        )
    out.mkdir(parents=True, exist_ok=False)
    ledger_path = out / "FINAL_TEST_ACCESS.json"
    started = {
        "version": VERSION,
        "dataset": DATASET,
        "primary_method": PRIMARY_METHOD,
        "status": "ARMED_BEFORE_TEST_CONSTRUCTION",
        "access_started_utc": utc_now(),
        "official_test_constructed": False,
        "official_test_cached_once": False,
        "rerun_allowed": False,
        "sample_level_test_artifacts_written": False,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
    }
    # Exclusive create is the one-shot guard.  There is intentionally no
    # overwrite flag in this evaluator.
    with ledger_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(started, indent=2, sort_keys=True) + "\n")

    try:
        # OFFICIAL TEST ACCESS STARTS HERE.  No development decision follows.
        test_loader = build_single_split_loader(args, "test", cli.num_workers)
        started["status"] = "TEST_CONSTRUCTED"
        started["official_test_constructed"] = True
        started["test_dataset_size"] = int(len(test_loader.dataset))
        json_write(ledger_path, started)

        batches, labels, indices = cache_official_test_once(test_loader)
        started["status"] = "TEST_CACHED_ONCE_IN_MEMORY"
        started["official_test_cached_once"] = True
        started["test_sample_count"] = int(len(labels))
        started["test_batch_count"] = int(len(batches))
        json_write(ledger_path, started)
        # Drop the dataset/loader owner.  All model inference now uses only the
        # in-memory batch cache and never iterates Official Test again.
        del test_loader

        member_predictions: Dict[int, Dict[str, np.ndarray]] = {}
        for seed in SEEDS:
            model = build_cfcompat_model(
                args, Path(sources["cfcompat"][seed]["checkpoint"])
            )
            member_predictions[seed] = infer_cached_batches(model, batches, args.device)
            release_model(model)

        model = build_v13_model(args, Path(sources["v13"]["checkpoint"]))
        v13_predictions = infer_cached_batches(model, batches, args.device)
        release_model(model)

        anchor, raw5, fixedblend, dp57, projection_summary = compose_frozen_method(
            member_predictions, v13_predictions
        )

        # Labels are first used here, after every fixed prediction and DP57
        # projection has already been produced.  Metrics cannot alter anything.
        metrics: Dict[str, Dict[str, Any]] = {}
        prediction_sets = {
            "Anchor": anchor,
            "Raw5": raw5,
            "v13": v13_predictions,
            "FixedBlend": fixedblend,
            "DP57": dp57,
        }
        rows = []
        for name, predictions in prediction_sets.items():
            local_metrics, j = evaluate_prediction_set(predictions, labels)
            metrics[name] = {"J": float(j), **local_metrics}
            rows.append(aggregate_row(name, local_metrics, j))

        aggregate_csv = out / "final_test_aggregate_metrics.csv"
        pd.DataFrame(rows).to_csv(aggregate_csv, index=False)

        summary = {
            "version": VERSION,
            "dataset": DATASET,
            "primary_method": PRIMARY_METHOD,
            "primary_row_name": "DP57",
            "test_sample_count": int(len(labels)),
            "aggregate_only": True,
            "sample_level_test_artifacts_written": False,
            "test_prediction_csv_written": False,
            "test_projection_row_csv_written": False,
            "test_ids_written": False,
            "test_labels_written": False,
            "protocol": {
                "final_test_access_count_intended": 1,
                "test_dataset_constructed_once": True,
                "test_dataset_iterated_once_into_cpu_memory": True,
                "model_inference_reuses_only_in_memory_test_batches": True,
                "raw5_members": list(SEEDS),
                "raw5_equal_weight": float(RAW5_MEMBER_WEIGHT),
                "v13_seed": V13_SEED,
                "blend_weights": {
                    "Raw5": float(BLEND_WEIGHT_RAW5),
                    "v13": float(BLEND_WEIGHT_V13),
                },
                "blend_weight_search_on_test": False,
                "model_selection_on_test": False,
                "anchor_selection_on_test": False,
                "anchor_seed": FROZEN_ANCHOR_SEED,
                "anchor_selected_on_valid": True,
                "dp_variant": DP_VARIANT,
                "dp_preserves": ["Acc7", "Acc5"],
                "labels_used_by_dp_projection": False,
                "labels_first_used_after_all_predictions_and_projection_frozen": True,
                "rerun_allowed_by_code": False,
            },
            "frozen_sources": sources,
            "metrics": metrics,
            "projection_summary": projection_summary,
            "environment": {
                "python_platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(args.device),
                "matmul_precision": cli.matmul_precision,
                "num_workers": int(cli.num_workers),
                "git_branch": git_value("branch", "--show-current"),
                "git_commit": git_value("rev-parse", "HEAD"),
                "completed_utc": utc_now(),
            },
        }
        summary_path = out / "final_test_summary.json"
        json_write(summary_path, summary)

        started.update(
            {
                "status": "COMPLETED",
                "completed_utc": utc_now(),
                "aggregate_metrics_csv": str(aggregate_csv),
                "aggregate_metrics_sha256": sha256(aggregate_csv),
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path),
            }
        )
        json_write(ledger_path, started)

        print("================ MOSEI FINAL TEST — ONE SHOT =====================")
        print("test samples:", len(labels))
        for name in ("Anchor", "Raw5", "v13", "FixedBlend", "DP57"):
            item = metrics[name]
            local = {key: value for key, value in item.items() if key != "J"}
            print(metric_line(name, local, float(item["J"])))
        for mode in MODES:
            item = projection_summary[mode]
            print(
                f"{mode} DP57 projection: projected={item['projected_fraction']:.6f} "
                f"feasible={item['target_feasible_fraction']:.6f} "
                f"boundary_adjusted={item['boundary_adjusted_count']} "
                f"fallback={item['fallback_count']} "
                f"acc2_changed_vs_blend={item['acc2_changed_vs_fixedblend_count']}"
            )
        print("primary method:", PRIMARY_METHOD)
        print("blend weight search on Test:", False)
        print("model/anchor selection on Test:", False)
        print("sample-level Test artifacts written:", False)
        print("TEST CONSTRUCTED:", True)
        print("FINAL TEST ACCESS LEDGER STATUS: COMPLETED")
        print("aggregate metrics:", aggregate_csv)
        print("summary:", summary_path)
        print("ledger:", ledger_path)

        # Explicitly release sample-level in-memory Test material before exit.
        del batches, labels, indices, member_predictions, v13_predictions
        del anchor, raw5, fixedblend, dp57
        gc.collect()
    except Exception as exc:
        started.update(
            {
                "status": "FAILED_AFTER_ONE_SHOT_WAS_ARMED",
                "failed_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "rerun_allowed": False,
            }
        )
        json_write(ledger_path, started)
        raise


def main() -> None:
    cli = parse_args()
    torch.cuda.set_device(int(cli.gpu_id)) if torch.cuda.is_available() else None
    torch.set_float32_matmul_precision(cli.matmul_precision)
    torch.backends.cudnn.allow_tf32 = cli.matmul_precision != "highest"
    if cli.preflight:
        run_preflight(cli)
    else:
        execute_final_test(cli)


if __name__ == "__main__":
    main()
