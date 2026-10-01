"""Validation-only compatibility-stratified distillation analysis.

This script never reads test predictions.

It compares three paired validation systems:
  1) ModDrop / No KD
  2) FixedKD / Uniform KD
  3) CFCompat / Compatibility-aware KD

For each validation sample and student condition (LA/LV/L), compatibility is
calibrated from the TRAIN-ONLY counterfactual delta distribution of the same
mode. Positive gains mean lower absolute error.

Main quantities:
  uniform_gain = error(NoKD) - error(FixedKD)
  compat_gain  = error(FixedKD) - error(CFCompat)
  total_gain   = error(NoKD) - error(CFCompat)
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_loader import MMDataLoader
from train_cf_compat_kd import build_config, prediction_rows
from trains.singleTask.missing_utils import MissingModalityWrapper
from trains.singleTask.model.DLF import DLF


MODES = ("LA", "LV", "L")
QUARTILE_LABELS = ("Q1_low", "Q2", "Q3", "Q4_high")


def parse_args():
    p = argparse.ArgumentParser(description="Validation-only compatibility-stratified KD analysis.")
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--result-root", default="result")
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--baseline-pred")
    p.add_argument("--fixedkd-pred")
    p.add_argument("--fixedkd-checkpoint")
    p.add_argument("--cfcompat-pred")
    p.add_argument("--train-cache")
    p.add_argument("--output-dir")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261001)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def default_paths(args):
    root = Path(args.result_root)
    baseline = Path(args.baseline_pred) if args.baseline_pred else (
        root / "missing_baseline" / "moddrop_benchmark_multiseed_v1"
        / f"seed{args.seed}" / f"{args.dataset}_best_valid_predictions.csv"
    )
    fixed_pred = Path(args.fixedkd_pred) if args.fixedkd_pred else None
    fixed_ckpt = Path(args.fixedkd_checkpoint) if args.fixedkd_checkpoint else (
        Path(args.model_save_dir) / "missing_baseline" / "fixed_kd"
        / f"DLF_{args.dataset}_seed{args.seed}_best.pth"
    )
    cfcompat = Path(args.cfcompat_pred) if args.cfcompat_pred else (
        root / "missing_baseline" / "cf_compat_kd_v1" / "benchmark_multiseed"
        / f"seed{args.seed}" / f"{args.dataset}_best_valid_predictions.csv"
    )
    cache = Path(args.train_cache) if args.train_cache else (
        root / "counterfactual_compatibility" / "cf_compat_v1_multiseed"
        / args.dataset / f"seed{args.seed}" / "train_counterfactual_compatibility.csv"
    )
    output = Path(args.output_dir) if args.output_dir else (
        root / "analysis" / "compatibility_gain_v2" / args.dataset / f"seed{args.seed}"
    )
    return baseline, fixed_pred, fixed_ckpt, cfcompat, cache, output


def forbid_test_path(path):
    if path is None:
        return
    text = str(path).replace("\\", "/").lower()
    if "test" in Path(path).name.lower() or "/test/" in text:
        raise ValueError(f"Validation-only analysis refuses test input: {path}")


def require_columns(frame, columns, path):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def generate_fixedkd_valid_predictions(args, checkpoint, output_path):
    """Generate per-sample validation predictions from the frozen FixedKD checkpoint only."""
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"FixedKD validation predictions were not provided and checkpoint is absent: {checkpoint}"
        )

    cli = SimpleNamespace(
        dataset=args.dataset,
        config_file=args.config_file,
        gpu_ids=args.gpu_ids,
    )
    cfg = build_config(cli, args.seed)
    loaders = MMDataLoader(cfg, args.num_workers)
    if "valid" not in loaders:
        raise RuntimeError("MMDataLoader did not expose a validation split.")

    backbone = DLF(cfg).to(cfg.device)
    model = MissingModalityWrapper(
        backbone, cfg.feature_dims[1], cfg.feature_dims[2]
    ).to(cfg.device)

    state = torch.load(checkpoint, map_location=cfg.device)
    model.load_state_dict(state, strict=True)
    frame = prediction_rows(model, loaders["valid"], cfg.device)
    frame["selected_by"] = "validation"
    frame["diagnostic_only"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def load_inputs(args, baseline_path, fixed_pred_path, fixed_ckpt, cfcompat_path, cache_path, output_dir):
    for path in (baseline_path, cfcompat_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        forbid_test_path(path)
    forbid_test_path(fixed_pred_path)
    forbid_test_path(fixed_ckpt)

    baseline = pd.read_csv(baseline_path)
    cfcompat = pd.read_csv(cfcompat_path)
    cache = pd.read_csv(cache_path)

    if fixed_pred_path is not None:
        if not fixed_pred_path.is_file():
            raise FileNotFoundError(fixed_pred_path)
        fixedkd = pd.read_csv(fixed_pred_path)
        fixed_source = str(fixed_pred_path)
    else:
        generated = output_dir / f"{args.dataset}_seed{args.seed}_fixedkd_valid_predictions.csv"
        if generated.is_file():
            fixedkd = pd.read_csv(generated)
        else:
            fixedkd = generate_fixedkd_valid_predictions(args, fixed_ckpt, generated)
        fixed_source = str(generated)

    pred_cols = ["sample_index", "sample_id", "label", "LAV_pred"] + [f"{m}_pred" for m in MODES]
    require_columns(baseline, pred_cols, baseline_path)
    require_columns(fixedkd, pred_cols, fixed_source)
    require_columns(cfcompat, pred_cols, cfcompat_path)
    require_columns(cache, ["sample_index"] + [f"delta_{m}" for m in MODES], cache_path)

    frames = {
        "baseline": baseline.sort_values("sample_index", kind="mergesort").reset_index(drop=True),
        "fixedkd": fixedkd.sort_values("sample_index", kind="mergesort").reset_index(drop=True),
        "cfcompat": cfcompat.sort_values("sample_index", kind="mergesort").reset_index(drop=True),
    }

    reference = frames["baseline"]
    for name, frame in frames.items():
        if frame.sample_index.duplicated().any():
            raise ValueError(f"{name} validation predictions contain duplicate sample_index.")
        if len(frame) != len(reference):
            raise ValueError(f"{name} validation prediction count differs from baseline.")
        for col in ("sample_index", "sample_id"):
            if not np.array_equal(reference[col].astype(str).to_numpy(), frame[col].astype(str).to_numpy()):
                raise ValueError(f"{name} is not paired with baseline by {col}.")
        if not np.allclose(reference["label"].to_numpy(float), frame["label"].to_numpy(float),
                           atol=1e-8, rtol=0):
            raise ValueError(f"{name} labels differ from baseline.")

    return frames["baseline"], frames["fixedkd"], frames["cfcompat"], cache, fixed_source


def train_calibrated_compatibility(train_delta, target_delta):
    """Map target deltas through the train empirical midpoint CDF."""
    train = np.sort(np.asarray(train_delta, dtype=np.float64))
    target = np.asarray(target_delta, dtype=np.float64)
    if len(train) == 0 or not np.isfinite(train).all() or np.any(train < 0):
        raise ValueError("Training deltas must be finite, non-negative, and non-empty.")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Target deltas must be finite and non-negative.")

    left = np.searchsorted(train, target, side="left").astype(np.float64)
    right = np.searchsorted(train, target, side="right").astype(np.float64)
    q = (left + 0.5 * (right - left)) / float(len(train))
    lo = 0.5 / float(len(train))
    hi = 1.0 - lo
    q = np.clip(q, lo, hi)
    return 1.0 - q


def build_long_frame(baseline, fixedkd, cfcompat, cache):
    rows = []
    labels = baseline["label"].to_numpy(dtype=np.float64)

    for mode in MODES:
        valid_delta = np.abs(
            baseline["LAV_pred"].to_numpy(dtype=np.float64)
            - baseline[f"{mode}_pred"].to_numpy(dtype=np.float64)
        )
        compat = train_calibrated_compatibility(
            cache[f"delta_{mode}"].to_numpy(dtype=np.float64),
            valid_delta,
        )

        e0 = np.abs(baseline[f"{mode}_pred"].to_numpy(dtype=np.float64) - labels)
        eu = np.abs(fixedkd[f"{mode}_pred"].to_numpy(dtype=np.float64) - labels)
        ec = np.abs(cfcompat[f"{mode}_pred"].to_numpy(dtype=np.float64) - labels)

        local = pd.DataFrame({
            "sample_index": baseline["sample_index"].to_numpy(),
            "sample_id": baseline["sample_id"].astype(str).to_numpy(),
            "label": labels,
            "mode": mode,
            "delta": valid_delta,
            "compatibility": compat,
            "nokd_abs_error": e0,
            "uniformkd_abs_error": eu,
            "cfcompat_abs_error": ec,
            "uniform_gain": e0 - eu,
            "compat_gain": eu - ec,
            "total_gain": e0 - ec,
            "uniform_better_than_nokd": eu < e0,
            "cfcompat_better_than_uniform": ec < eu,
            "cfcompat_better_than_nokd": ec < e0,
            "uniform_negative_transfer": eu > e0,
            "recovered_vs_uniform": (eu > e0) & (ec < eu),
            "fully_recovered_vs_nokd": (eu > e0) & (ec <= e0),
        })
        rows.append(local)

    frame = pd.concat(rows, ignore_index=True)
    frame["compat_quartile"] = pd.cut(
        frame["compatibility"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=QUARTILE_LABELS,
        include_lowest=True,
        right=True,
    )
    if frame["compat_quartile"].isna().any():
        raise RuntimeError("Some compatibility values were not assigned to quartiles.")
    return frame


def cluster_bootstrap_ci(frame, value_col, reps, seed):
    """Bootstrap by sample_index so LA/LV/L observations from one sample stay clustered."""
    ids = frame["sample_index"].drop_duplicates().to_numpy()
    if len(ids) < 2:
        return np.nan, np.nan
    grouped = {
        idx: frame.loc[frame.sample_index == idx, value_col].to_numpy(float)
        for idx in ids
    }
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        values = np.concatenate([grouped[idx] for idx in sampled])
        means[r] = float(values.mean())
    return tuple(np.quantile(means, [0.025, 0.975]))


def aggregate_group(frame, group_type, group_name, reps, seed):
    u_lo, u_hi = cluster_bootstrap_ci(frame, "uniform_gain", reps, seed)
    c_lo, c_hi = cluster_bootstrap_ci(frame, "compat_gain", reps, seed + 1)
    t_lo, t_hi = cluster_bootstrap_ci(frame, "total_gain", reps, seed + 2)

    return {
        "group_type": group_type,
        "group": group_name,
        "count_rows": int(len(frame)),
        "count_samples": int(frame.sample_index.nunique()),
        "mean_compatibility": float(frame.compatibility.mean()),
        "mean_delta": float(frame.delta.mean()),
        "nokd_mae": float(frame.nokd_abs_error.mean()),
        "uniformkd_mae": float(frame.uniformkd_abs_error.mean()),
        "cfcompat_mae": float(frame.cfcompat_abs_error.mean()),
        "uniform_gain": float(frame.uniform_gain.mean()),
        "uniform_gain_ci95_low": float(u_lo),
        "uniform_gain_ci95_high": float(u_hi),
        "compat_gain": float(frame.compat_gain.mean()),
        "compat_gain_ci95_low": float(c_lo),
        "compat_gain_ci95_high": float(c_hi),
        "total_gain": float(frame.total_gain.mean()),
        "total_gain_ci95_low": float(t_lo),
        "total_gain_ci95_high": float(t_hi),
        "uniform_improved_rate": float(frame.uniform_better_than_nokd.mean()),
        "compat_improved_rate": float(frame.cfcompat_better_than_uniform.mean()),
        "total_improved_rate": float(frame.cfcompat_better_than_nokd.mean()),
    }


def summarize(frame, reps, seed):
    rows = [aggregate_group(frame, "overall", "all", reps, seed)]

    for j, label in enumerate(QUARTILE_LABELS):
        local = frame.loc[frame.compat_quartile.astype(str) == label]
        rows.append(aggregate_group(local, "compat_quartile", label, reps, seed + 100 + j * 5))

    for j, mode in enumerate(MODES):
        local = frame.loc[frame["mode"] == mode]
        rows.append(aggregate_group(local, "mode", mode, reps, seed + 200 + j * 20))
        for k, label in enumerate(QUARTILE_LABELS):
            cell = local.loc[local.compat_quartile.astype(str) == label]
            rows.append(aggregate_group(
                cell, f"mode_{mode}_quartile", label, reps, seed + 300 + j * 50 + k * 5
            ))

    summary = pd.DataFrame(rows)

    def corr(col):
        pearson = float(frame["compatibility"].corr(frame[col], method="pearson"))
        spearman = float(
            frame["compatibility"].rank(method="average").corr(
                frame[col].rank(method="average"), method="pearson"
            )
        )
        return pearson, spearman

    p_u, s_u = corr("uniform_gain")
    p_c, s_c = corr("compat_gain")
    p_t, s_t = corr("total_gain")

    q = summary.loc[summary.group_type == "compat_quartile"].set_index("group")
    diagnostics = {
        "split": "validation_only",
        "compatibility_definition":
            "1 - train empirical midpoint CDF of validation counterfactual delta, calibrated independently per mode",
        "uniform_gain_definition":
            "NoKD absolute error - FixedKD absolute error; positive means uniform KD helps",
        "compat_gain_definition":
            "FixedKD absolute error - CFCompat absolute error; positive means compatibility-aware KD improves over uniform KD",
        "total_gain_definition":
            "NoKD absolute error - CFCompat absolute error; positive means CFCompat improves over NoKD",
        "pearson_compatibility_uniform_gain": p_u,
        "spearman_compatibility_uniform_gain": s_u,
        "pearson_compatibility_compat_gain": p_c,
        "spearman_compatibility_compat_gain": s_c,
        "pearson_compatibility_total_gain": p_t,
        "spearman_compatibility_total_gain": s_t,
        "q1_uniform_gain": float(q.loc["Q1_low", "uniform_gain"]),
        "q4_uniform_gain": float(q.loc["Q4_high", "uniform_gain"]),
        "q1_compat_gain": float(q.loc["Q1_low", "compat_gain"]),
        "q4_compat_gain": float(q.loc["Q4_high", "compat_gain"]),
        "q1_total_gain": float(q.loc["Q1_low", "total_gain"]),
        "q4_total_gain": float(q.loc["Q4_high", "total_gain"]),
    }
    return summary, diagnostics



def negative_transfer_analysis(frame):
    """Summarize how often CFCompat mitigates harmful uniform KD."""
    rows = []
    groups = [("overall", "all", frame)]
    groups.extend(("mode", mode, frame.loc[frame["mode"] == mode]) for mode in MODES)

    for group_type, group_name, local in groups:
        negative = local.loc[local["uniform_negative_transfer"]].copy()
        row = {
            "group_type": group_type,
            "group": group_name,
            "count_rows": int(len(local)),
            "negative_transfer_count": int(len(negative)),
            "negative_transfer_rate": float(len(negative) / len(local)) if len(local) else np.nan,
            "recovery_rate_vs_uniform": np.nan,
            "full_recovery_rate_vs_nokd": np.nan,
            "mean_uniform_harm": np.nan,
            "mean_cfcompat_recovery": np.nan,
            "mean_net_gain_after_cfcompat": np.nan,
        }
        if len(negative):
            row.update({
                "recovery_rate_vs_uniform": float(negative["recovered_vs_uniform"].mean()),
                "full_recovery_rate_vs_nokd": float(negative["fully_recovered_vs_nokd"].mean()),
                "mean_uniform_harm": float((-negative["uniform_gain"]).mean()),
                "mean_cfcompat_recovery": float(negative["compat_gain"].mean()),
                "mean_net_gain_after_cfcompat": float(negative["total_gain"].mean()),
            })
        rows.append(row)

    return pd.DataFrame(rows)


def make_plot(summary, output_path):
    import matplotlib.pyplot as plt

    q = summary.loc[summary.group_type == "compat_quartile"].copy()
    order = {name: i for i, name in enumerate(QUARTILE_LABELS)}
    q["order"] = q["group"].map(order)
    q = q.sort_values("order")

    x = np.arange(len(q))
    uniform = q["uniform_gain"].to_numpy(float)
    compat = q["compat_gain"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.plot(x, uniform, marker="s", linestyle="--", linewidth=1.8, label="Uniform KD vs No KD")
    ax.plot(x, compat, marker="o", linestyle="-", linewidth=1.8, label="CFCompat vs Uniform KD")
    ax.axhline(0.0, linewidth=1.0, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Q2", "Q3", "High"])
    ax.set_xlabel("Compatibility quartile")
    ax.set_ylabel("MAE improvement")
    ax.set_title("Compatibility-stratified Distillation Effect")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    baseline_path, fixed_pred_path, fixed_ckpt, cfcompat_path, cache_path, output_dir = default_paths(args)
    baseline, fixedkd, cfcompat, cache, fixed_source = load_inputs(
        args, baseline_path, fixed_pred_path, fixed_ckpt, cfcompat_path, cache_path, output_dir
    )
    frame = build_long_frame(baseline, fixedkd, cfcompat, cache)
    summary, diagnostics = summarize(frame, args.bootstrap_reps, args.bootstrap_seed)
    mitigation = negative_transfer_analysis(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_distillation_effect_samples.csv"
    summary_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_distillation_effect_summary.csv"
    json_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_distillation_effect_diagnostics.json"
    plot_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_distillation_effect.png"
    mitigation_path = output_dir / f"{args.dataset}_seed{args.seed}_negative_transfer_mitigation.csv"

    frame.to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)
    mitigation.to_csv(mitigation_path, index=False)
    diagnostics["baseline_predictions"] = str(baseline_path)
    diagnostics["fixedkd_predictions"] = fixed_source
    diagnostics["cfcompat_predictions"] = str(cfcompat_path)
    diagnostics["train_cache"] = str(cache_path)
    json_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plot:
        make_plot(summary, plot_path)

    display_cols = [
        "group_type", "group", "count_rows", "mean_compatibility",
        "nokd_mae", "uniformkd_mae", "cfcompat_mae",
        "uniform_gain", "compat_gain", "total_gain",
        "uniform_improved_rate", "compat_improved_rate"
    ]

    print("Validation-only compatibility-stratified KD analysis complete.")
    print(f"baseline={baseline_path}")
    print(f"fixedkd={fixed_source}")
    print(f"cfcompat={cfcompat_path}")
    print(f"train_cache={cache_path}")
    print(f"summary={summary_path}")
    print(f"diagnostics={json_path}")
    print(f"negative_transfer={mitigation_path}")
    if not args.no_plot:
        print(f"plot={plot_path}")
    print()
    print(summary.loc[
        summary.group_type.isin(["overall", "compat_quartile"]), display_cols
    ].to_string(index=False))
    print()
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print()
    print("Negative-transfer mitigation:")
    print(mitigation.to_string(index=False))


if __name__ == "__main__":
    main()
