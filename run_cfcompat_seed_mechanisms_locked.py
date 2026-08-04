"""Locked efficient entrypoint for the CFCompatKD mechanism audit.

It replaces only the gradient extraction loop with a three-backward implementation
over the union of the frozen parameter groups. Outputs and protocol are unchanged.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import analyze_cfcompat_seed_mechanisms as audit_main
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cfcompat_mechanism_utils import (
    FORMAL_SEEDS,
    MISSING_MODES,
    OBJECTIVE_PAIRS,
    gradient_pair_stats,
    parameter_groups,
    tensor_state_sha256,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from utils.functions import setup_seed


def gradient_audit_fast(cli, source, output_root):
    gradient_rows = []
    state_rows = []
    group_manifest_rows = []
    probe_reference = None

    for seed in FORMAL_SEEDS:
        setup_seed(seed)
        args = audit_main.build_config(cli, seed)
        probe_loader, probe_manifest = audit_main.build_probe(
            cli, args, output_root
        )
        current_probe = probe_manifest[
            ["sample_index", "sample_id", "label", "video_id"]
        ].copy()
        if probe_reference is None:
            probe_reference = current_probe
        elif not current_probe.equals(probe_reference):
            raise RuntimeError("Probe changed across formal seeds.")

        variants = (
            ("baseline", "adam_replay"),
            ("sam", source["selected_run"]),
        )
        for variant, run in variants:
            row = audit_main.source_row(source, seed, run)
            checkpoint = audit_main.resolve_path(
                row["MainCheckpoint"], cli.result_root
            )
            model = audit_main.load_student(args, checkpoint)
            teacher, cache_by_index = audit_main.load_train_assets(
                cli, args, row
            )
            model.train()
            groups = parameter_groups(model)
            group_items = list(groups.items())
            flat_parameters = []
            group_slices = {}
            for group_name, named_params in group_items:
                start = len(flat_parameters)
                flat_parameters.extend(
                    parameter for _, parameter in named_params
                )
                group_slices[group_name] = slice(
                    start, len(flat_parameters)
                )
                group_manifest_rows.append({
                    "Seed": int(seed),
                    "Variant": variant,
                    "ParameterGroup": group_name,
                    "parameter_count": int(len(named_params)),
                    "parameter_numel": int(
                        sum(parameter.numel() for _, parameter in named_params)
                    ),
                })

            before_sha = tensor_state_sha256(model)
            criterion = nn.L1Loss()
            cosine = nn.CosineEmbeddingLoss()
            hinge = HingeLoss()

            for batch_index, batch in enumerate(probe_loader):
                for mode in MISSING_MODES:
                    losses = audit_main.objective_losses(
                        model,
                        teacher,
                        cache_by_index,
                        batch,
                        mode,
                        args,
                        rng_seed=(
                            91000000
                            + int(seed) * 100
                            + int(batch_index)
                        ),
                        criterion=criterion,
                        cosine=cosine,
                        hinge=hinge,
                    )
                    gradients = {
                        "full": torch.autograd.grad(
                            losses["full"],
                            flat_parameters,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True,
                        ),
                        "missing": torch.autograd.grad(
                            losses["missing"],
                            flat_parameters,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True,
                        ),
                        "kd": torch.autograd.grad(
                            losses["kd"],
                            flat_parameters,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        ),
                    }

                    for group_name, _ in group_items:
                        local_slice = group_slices[group_name]
                        for left, right in OBJECTIVE_PAIRS:
                            stats = gradient_pair_stats(
                                gradients[left][local_slice],
                                gradients[right][local_slice],
                            )
                            gradient_rows.append({
                                "Seed": int(seed),
                                "Variant": variant,
                                "Run": run,
                                "Mode": mode,
                                "ProbeBatch": int(batch_index),
                                "ParameterGroup": group_name,
                                "Pair": "{}_vs_{}".format(left, right),
                                "full_loss": float(
                                    losses["full"].detach().cpu()
                                ),
                                "missing_loss": float(
                                    losses["missing"].detach().cpu()
                                ),
                                "kd_loss": float(
                                    losses["kd"].detach().cpu()
                                ),
                                **stats,
                            })

                    del gradients, losses
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            after_sha = tensor_state_sha256(model)
            state_rows.append({
                "Seed": int(seed),
                "Variant": variant,
                "Run": run,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256(checkpoint),
                "state_sha_before": before_sha,
                "state_sha_after": after_sha,
                "parameters_unchanged": bool(before_sha == after_sha),
            })
            del model, teacher, flat_parameters, group_slices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del probe_loader

    raw = pd.DataFrame(gradient_rows)
    raw.to_csv(output_root / "gradient_pairwise_raw.csv", index=False)
    pd.DataFrame(state_rows).to_csv(
        output_root / "gradient_model_state_audit.csv", index=False
    )
    pd.DataFrame(group_manifest_rows).drop_duplicates().to_csv(
        output_root / "gradient_parameter_groups.csv", index=False
    )

    summary_rows = []
    for keys, local in raw.groupby(
        ["Seed", "Variant", "Mode", "ParameterGroup", "Pair"],
        sort=True,
    ):
        finite_cos = local.cosine[np.isfinite(local.cosine.astype(float))]
        summary_rows.append({
            "Seed": int(keys[0]),
            "Variant": str(keys[1]),
            "Mode": str(keys[2]),
            "ParameterGroup": str(keys[3]),
            "Pair": str(keys[4]),
            "batch_count": int(len(local)),
            "finite_cosine_count": int(len(finite_cos)),
            "mean_cosine": (
                float(finite_cos.astype(float).mean())
                if len(finite_cos) else float("nan")
            ),
            "median_cosine": (
                float(finite_cos.astype(float).median())
                if len(finite_cos) else float("nan")
            ),
            "conflict_fraction": (
                float((finite_cos.astype(float) < 0).mean())
                if len(finite_cos) else float("nan")
            ),
            "mean_left_norm": float(local.left_norm.mean()),
            "mean_right_norm": float(local.right_norm.mean()),
            "all_finite": bool(local.finite.all()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        output_root / "gradient_pair_summary.csv", index=False
    )

    baseline = summary.loc[
        summary.Variant.eq("baseline")
    ].drop(columns=["Variant"])
    sam = summary.loc[
        summary.Variant.eq("sam")
    ].drop(columns=["Variant"])
    keys = ["Seed", "Mode", "ParameterGroup", "Pair"]
    delta = baseline.merge(
        sam,
        on=keys,
        suffixes=("_baseline", "_sam"),
        validate="one_to_one",
    )
    delta["mean_cosine_change_sam_minus_baseline"] = (
        delta.mean_cosine_sam - delta.mean_cosine_baseline
    )
    delta["conflict_fraction_change_sam_minus_baseline"] = (
        delta.conflict_fraction_sam - delta.conflict_fraction_baseline
    )
    delta.to_csv(
        output_root / "gradient_sam_minus_baseline.csv", index=False
    )
    return {
        "raw": raw,
        "summary": summary,
        "delta": delta,
        "state": pd.DataFrame(state_rows),
    }


if __name__ == "__main__":
    audit_main.gradient_audit = gradient_audit_fast
    audit_main.main()
