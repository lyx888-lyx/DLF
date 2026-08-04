"""One permitted repair for CFCompatKD C-Mixup: mix the full LAV view only."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

import train_cfcompat_cmixup as base
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_COLUMNS,
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    cache_paths,
)
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.cmixup_regression_utils import (
    mix_fusion_features,
    predict_from_dlf_fusion,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES


VERSION = "cfcompat_cmixup_full_only_v2"
METHOD = "DLF-CFCompatKD-FullViewCMixup-v2"
OUTPUT_TAG = "cfcompat_cmixup_full_only_v2"


def portable_locate_stage1_evaluator(
    result_root, dataset, seed, multiseed=False, smoke=False
):
    """Resolve historical CSV checkpoint paths against the old project root."""
    if multiseed:
        source = (
            Path(result_root)
            / "missing_baseline"
            / "moddrop_benchmark_multiseed_v1"
        )
        if smoke:
            source = source / "smoke"
        source = (
            source
            / "seed{}".format(int(seed))
            / "{}_per_seed.csv".format(dataset)
        )
        checkpoint_field, epoch_field = "MainCheckpoint", "BestValidEpoch"
    else:
        source = (
            Path(result_root)
            / "missing_baseline"
            / "moddrop"
            / "train"
            / "{}_per_seed.csv".format(dataset)
        )
        checkpoint_field, epoch_field = "Checkpoint", "BestEpoch"

    if not source.is_file():
        raise FileNotFoundError(
            "Required Stage 1 result CSV absent: {}".format(source)
        )
    rows = pd.read_csv(source)
    selected = rows.loc[rows.Seed.astype(int).eq(int(seed))]
    if len(selected) != 1 or checkpoint_field not in selected:
        raise ValueError(
            "Stage 1 CSV has no unique checkpoint for seed {}.".format(seed)
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
        raise FileNotFoundError(
            "Stage 1 CSV checkpoint is absent: {}".format(checkpoint)
        )
    return checkpoint, int(selected.iloc[0][epoch_field]), source


def load_locked_counterfactual_cache(
    root,
    dataset,
    version=CACHE_VERSION,
    seed=None,
    expected_evaluator_sha=None,
):
    """Load new caches and the known locked seed1111 historical format."""
    paths = cache_paths(root, dataset, version=version, seed=seed)
    if not paths["csv"].is_file() or not paths["config"].is_file():
        raise FileNotFoundError("Counterfactual cache has not been built.")

    frame = pd.read_csv(paths["csv"])
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    expected_seed = None if seed is None else int(seed)

    if config.get("version") != version or config.get("seed") != expected_seed:
        raise ValueError("Counterfactual cache version/seed binding is invalid.")
    if (
        expected_evaluator_sha is not None
        and config.get("evaluator_sha256") != expected_evaluator_sha
    ):
        raise ValueError("Counterfactual cache evaluator SHA binding is invalid.")
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
        raise ValueError("Counterfactual cache sample indices are duplicated.")
    if len(frame) != 1284 or frame.sample_index.nunique() != 1284:
        raise ValueError("MOSI train cache must contain 1284 unique samples.")

    for mode in MISSING_MODES:
        values = frame["compat_{}".format(mode)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.all(
            (values > 0) & (values < 1)
        ):
            raise ValueError("Cached compatibility is outside (0,1).")

    by_index = {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }
    return frame, by_index


def full_only_cmixup_loss(
    student: torch.nn.Module,
    full_fusion: torch.Tensor,
    missing_fusion: torch.Tensor,
    labels: torch.Tensor,
    modes: Sequence[str],
    sampler,
    criterion: torch.nn.Module,
):
    """Apply label-neighbour Mixup only to the complete LAV fusion feature."""
    del modes
    if full_fusion.ndim != 2 or full_fusion.size(0) != labels.size(0):
        raise ValueError("Full fusion feature/label shape mismatch.")
    if missing_fusion.shape != full_fusion.shape:
        raise ValueError("Missing fusion capture shape mismatch.")

    # V1 grouped the LAV partner pool by a random missing-mode draw solely
    # because it also mixed the missing representation. V2 has no missing
    # mixed loss, so all complete-view samples use one label-KDE pool.
    full_pool_modes = ["L"] * int(labels.size(0))
    batch = sampler.sample(
        labels=labels,
        modes=full_pool_modes,
        device=full_fusion.device,
        dtype=full_fusion.dtype,
    )

    if not bool(batch.active_mask.any()):
        zero = full_fusion.sum() * 0.0
        diagnostics = {
            "mix_active_fraction": 0.0,
            "mix_mean_lambda": 1.0,
            "mix_mean_partner_label_distance": 0.0,
            "mix_full_loss": 0.0,
            "mix_missing_loss": 0.0,
            "mix_partner_sha256": batch.partner_sha256,
            "mix_LA_count": 0,
            "mix_LV_count": 0,
            "mix_L_count": int(labels.size(0)),
        }
        return zero, diagnostics, batch

    mixed_full = mix_fusion_features(
        full_fusion, batch.partner_indices, batch.lambdas
    )
    with preserve_rng_state():
        prediction = predict_from_dlf_fusion(student.backbone, mixed_full)

    mask = batch.active_mask
    full_loss = criterion(prediction[mask], batch.mixed_labels[mask])
    diagnostics = {
        "mix_active_fraction": batch.active_fraction,
        "mix_mean_lambda": batch.mean_lambda,
        "mix_mean_partner_label_distance": batch.mean_partner_label_distance,
        "mix_full_loss": float(full_loss.detach()),
        "mix_missing_loss": 0.0,
        "mix_partner_sha256": batch.partner_sha256,
        "mix_LA_count": 0,
        "mix_LV_count": 0,
        "mix_L_count": int(labels.size(0)),
    }
    return full_loss, diagnostics, batch


def v2_paths(cli):
    result = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "seed{}".format(cli.seed)
    )
    model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "seed{}".format(cli.seed)
    )
    if cli.smoke_test:
        result, model = result / "smoke", model / "smoke"
    result.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return result, model / "best_valid.pth"


def render_v2_report(summary):
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    gate = summary["promotion_gate"]
    missing = MISSING_MODES
    baseline_missing = float(
        np.mean([baseline["valid_{}_MAE".format(mode)] for mode in missing])
    )
    candidate_missing = float(
        np.mean([candidate["valid_{}_MAE".format(mode)] for mode in missing])
    )
    lines = [
        "# CFCompatKD + Full-View C-Mixup v2",
        "",
        "## Decision",
        "",
        "- Verdict: `{}`".format(summary["verdict"]),
        "- C-Mixup scope: complete LAV fusion only",
        "- Missing-view mixed-label loss: disabled",
        "- Partner pool: whole training batch, label-KDE sampled",
        "",
        "## Valid metrics",
        "",
        "| Metric | CFCompatKD | V2 | Gain |",
        "| --- | ---: | ---: | ---: |",
        "| Valid J | {:.6f} | {:.6f} | {:+.6f} |".format(
            baseline["J_valid"], candidate["J_valid"], gate["gain_valid_J"]
        ),
        "| Valid LAV MAE | {:.6f} | {:.6f} | {:+.6f} |".format(
            baseline["valid_LAV_MAE"],
            candidate["valid_LAV_MAE"],
            gate["gain_valid_LAV_MAE"],
        ),
        "| Valid MissingMacro | {:.6f} | {:.6f} | {:+.6f} |".format(
            baseline_missing,
            candidate_missing,
            baseline_missing - candidate_missing,
        ),
        "",
        "## Hard stop",
        "",
        "If this seed fails the unchanged promotion gate, C-Mixup stops; no "
        "hyperparameter, layer, warmup, or partner-policy search is allowed.",
    ]
    return "\n".join(lines)


def patch_base_module():
    base.VERSION = VERSION
    base.METHOD = METHOD
    base.paths = v2_paths
    base.locate_stage1_evaluator = portable_locate_stage1_evaluator
    base.load_counterfactual_cache = load_locked_counterfactual_cache
    base.compute_mode_consistent_cmixup_loss = full_only_cmixup_loss


def main():
    patch_base_module()
    cli = base.parse_args()
    logger, log_path = base.create_logger(cli)
    result_dir, summary = base.train_one_seed(cli, logger)

    summary["version"] = VERSION
    summary["method"] = METHOD
    summary["candidate"]["Method"] = METHOD
    summary["candidate"]["CMixupScope"] = "LAV_fusion_only"
    summary["candidate"]["MixedMissingViewLoss"] = False
    summary["protocol"].update(
        {
            "fusion_location": "DLF.proj1 forward-pre-hook input",
            "full_view_only_cmixup": True,
            "whole_batch_label_kde_partner_pool": True,
            "same_partner_full_and_missing": False,
            "same_missing_mode_only": False,
            "mixed_missing_view_loss": False,
            "mixed_samples_use_teacher_or_compatibility": False,
            "original_missing_task_and_cfcompat_losses_unchanged": True,
            "additional_inference_parameters": 0,
        }
    )

    (result_dir / "cmixup_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (result_dir / "cmixup_report.md").write_text(
        render_v2_report(summary) + "\n", encoding="utf-8"
    )

    source_path = result_dir / "cmixup_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source.update(
        {
            "version": VERSION,
            "method": METHOD,
            "cmixup_scope": "LAV_fusion_only",
            "mixed_missing_view_loss": False,
            "candidate_checkpoint_sha256": checkpoint_sha256(
                Path(source["candidate_checkpoint"])
            ),
        }
    )
    source_path.write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "v2 result=%s verdict=%s report=%s log=%s",
        result_dir,
        summary["verdict"],
        result_dir / "cmixup_report.md",
        log_path,
    )
    print("CFCompat full-view C-Mixup v2 complete")
    print("seed:", cli.seed)
    print("verdict:", summary["verdict"])
    print(
        "valid J gain:",
        "{:+.6f}".format(summary["promotion_gate"]["gain_valid_J"]),
    )
    print(
        "valid LAV MAE gain:",
        "{:+.6f}".format(
            summary["promotion_gate"]["gain_valid_LAV_MAE"]
        ),
    )
    print("report:", result_dir / "cmixup_report.md")


if __name__ == "__main__":
    main()
