"""Validation-only analysis of whether CFCompat compatibility predicts KD benefit.

This script never reads test predictions. It:
1) calibrates validation counterfactual deltas against the train-only compatibility cache,
2) compares paired ModDrop vs CFCompat validation absolute errors, and
3) summarizes improvement by compatibility quartile.

Positive gain means CFCompat reduces absolute error relative to ModDrop.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("LA", "LV", "L")
QUARTILE_LABELS = ("Q1_low", "Q2", "Q3", "Q4_high")


def parse_args():
    p = argparse.ArgumentParser(description="Validation-only CFCompat compatibility/gain analysis.")
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--result-root", default="result")
    p.add_argument("--baseline-pred")
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
    cfcompat = Path(args.cfcompat_pred) if args.cfcompat_pred else (
        root / "missing_baseline" / "cf_compat_kd_v1" / "benchmark_multiseed"
        / f"seed{args.seed}" / f"{args.dataset}_best_valid_predictions.csv"
    )
    cache = Path(args.train_cache) if args.train_cache else (
        root / "counterfactual_compatibility" / "cf_compat_v1_multiseed"
        / args.dataset / f"seed{args.seed}" / "train_counterfactual_compatibility.csv"
    )
    output = Path(args.output_dir) if args.output_dir else (
        root / "analysis" / "compatibility_gain_v1" / args.dataset / f"seed{args.seed}"
    )
    return baseline, cfcompat, cache, output


def forbid_test_path(path):
    text = str(path).replace("\\", "/").lower()
    if "test" in Path(path).name.lower() or "/test/" in text:
        raise ValueError(f"Validation-only analysis refuses test input: {path}")


def require_columns(frame, columns, path):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def load_inputs(baseline_path, cfcompat_path, cache_path):
    for path in (baseline_path, cfcompat_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    forbid_test_path(baseline_path)
    forbid_test_path(cfcompat_path)

    baseline = pd.read_csv(baseline_path)
    cfcompat = pd.read_csv(cfcompat_path)
    cache = pd.read_csv(cache_path)

    pred_cols = ["sample_index", "sample_id", "label", "LAV_pred"] + [f"{m}_pred" for m in MODES]
    require_columns(baseline, pred_cols, baseline_path)
    require_columns(cfcompat, pred_cols, cfcompat_path)
    require_columns(cache, ["sample_index"] + [f"delta_{m}" for m in MODES], cache_path)

    baseline = baseline.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    cfcompat = cfcompat.sort_values("sample_index", kind="mergesort").reset_index(drop=True)

    if baseline.sample_index.duplicated().any() or cfcompat.sample_index.duplicated().any():
        raise ValueError("Validation prediction files must contain one row per sample_index.")
    if len(baseline) != len(cfcompat):
        raise ValueError("Baseline and CFCompat validation prediction counts differ.")

    for col in ("sample_index", "sample_id"):
        if not np.array_equal(baseline[col].astype(str).to_numpy(), cfcompat[col].astype(str).to_numpy()):
            raise ValueError(f"Baseline and CFCompat are not paired by {col}.")
    if not np.allclose(baseline["label"].to_numpy(float), cfcompat["label"].to_numpy(float), atol=1e-8, rtol=0):
        raise ValueError("Baseline and CFCompat labels differ.")

    return baseline, cfcompat, cache


def train_calibrated_compatibility(train_delta, target_delta):
    """Map target deltas through the train empirical midpoint CDF.

    For a training value this matches q=(average_rank-0.5)/N:
      q = (# train < d + 0.5 * # train == d) / N
      C = 1 - q

    Values outside the train support are clipped to the training score range.
    """
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


def build_long_frame(baseline, cfcompat, cache):
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
        base_error = np.abs(baseline[f"{mode}_pred"].to_numpy(dtype=np.float64) - labels)
        cf_error = np.abs(cfcompat[f"{mode}_pred"].to_numpy(dtype=np.float64) - labels)
        gain = base_error - cf_error

        local = pd.DataFrame({
            "sample_index": baseline["sample_index"].to_numpy(),
            "sample_id": baseline["sample_id"].astype(str).to_numpy(),
            "label": labels,
            "mode": mode,
            "delta": valid_delta,
            "compatibility": compat,
            "baseline_abs_error": base_error,
            "cfcompat_abs_error": cf_error,
            "gain": gain,
            "cfcompat_better": gain > 0,
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


def cluster_bootstrap_ci(frame, reps, seed):
    """Bootstrap mean gain by sample_index so repeated modes stay clustered."""
    ids = frame["sample_index"].drop_duplicates().to_numpy()
    if len(ids) < 2:
        return np.nan, np.nan
    grouped = {idx: frame.loc[frame.sample_index == idx, "gain"].to_numpy(float) for idx in ids}
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        values = np.concatenate([grouped[idx] for idx in sampled])
        means[r] = float(values.mean())
    return tuple(np.quantile(means, [0.025, 0.975]))


def aggregate_group(frame, group_type, group_name, reps, seed):
    lo, hi = cluster_bootstrap_ci(frame, reps, seed)
    return {
        "group_type": group_type,
        "group": group_name,
        "count_rows": int(len(frame)),
        "count_samples": int(frame.sample_index.nunique()),
        "mean_compatibility": float(frame.compatibility.mean()),
        "mean_delta": float(frame.delta.mean()),
        "baseline_mae": float(frame.baseline_abs_error.mean()),
        "cfcompat_mae": float(frame.cfcompat_abs_error.mean()),
        "mean_gain": float(frame.gain.mean()),
        "gain_ci95_low": float(lo),
        "gain_ci95_high": float(hi),
        "improved_rate": float(frame.cfcompat_better.mean()),
    }


def summarize(frame, reps, seed):
    rows = [aggregate_group(frame, "overall", "all", reps, seed)]

    for j, label in enumerate(QUARTILE_LABELS):
        local = frame.loc[frame.compat_quartile.astype(str) == label]
        rows.append(aggregate_group(local, "compat_quartile", label, reps, seed + 100 + j))

    for j, mode in enumerate(MODES):
        local = frame.loc[frame["mode"] == mode]
        rows.append(aggregate_group(local, "mode", mode, reps, seed + 200 + j))
        for k, label in enumerate(QUARTILE_LABELS):
            cell = local.loc[local.compat_quartile.astype(str) == label]
            rows.append(aggregate_group(cell, f"mode_{mode}_quartile", label, reps, seed + 300 + j * 10 + k))

    summary = pd.DataFrame(rows)

    pearson = float(frame["compatibility"].corr(frame["gain"], method="pearson"))
    spearman = float(frame["compatibility"].rank(method="average").corr(
        frame["gain"].rank(method="average"), method="pearson"
    ))
    q = summary.loc[summary.group_type == "compat_quartile"].set_index("group")
    high_minus_low = float(q.loc["Q4_high", "mean_gain"] - q.loc["Q1_low", "mean_gain"])

    diagnostics = {
        "pearson_compatibility_gain": pearson,
        "spearman_compatibility_gain": spearman,
        "q4_minus_q1_mean_gain": high_minus_low,
        "gain_definition": "baseline_abs_error - cfcompat_abs_error; positive means CFCompat improves",
        "compatibility_definition": "1 - train empirical midpoint CDF of validation counterfactual delta, calibrated independently per mode",
        "split": "validation_only",
    }
    return summary, diagnostics


def make_plot(summary, output_path):
    import matplotlib.pyplot as plt

    q = summary.loc[summary.group_type == "compat_quartile"].copy()
    q["order"] = q["group"].map({name: i for i, name in enumerate(QUARTILE_LABELS)})
    q = q.sort_values("order")

    x = np.arange(len(q))
    y = q["mean_gain"].to_numpy(float)
    lower = y - q["gain_ci95_low"].to_numpy(float)
    upper = q["gain_ci95_high"].to_numpy(float) - y

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=4, linewidth=1.8)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Q2", "Q3", "High"])
    ax.set_xlabel("Compatibility quartile")
    ax.set_ylabel("MAE improvement over ModDrop")
    ax.set_title("Does Compatibility Predict Distillation Benefit?")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    baseline_path, cfcompat_path, cache_path, output_dir = default_paths(args)
    baseline, cfcompat, cache = load_inputs(baseline_path, cfcompat_path, cache_path)
    frame = build_long_frame(baseline, cfcompat, cache)
    summary, diagnostics = summarize(frame, args.bootstrap_reps, args.bootstrap_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_compatibility_gain_samples.csv"
    summary_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_compatibility_gain_summary.csv"
    json_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_compatibility_gain_diagnostics.json"
    plot_path = output_dir / f"{args.dataset}_seed{args.seed}_valid_compatibility_gain.png"

    frame.to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plot:
        make_plot(summary, plot_path)

    print("Validation-only compatibility/gain analysis complete.")
    print(f"baseline={baseline_path}")
    print(f"cfcompat={cfcompat_path}")
    print(f"train_cache={cache_path}")
    print(f"sample_rows={sample_path}")
    print(f"summary={summary_path}")
    print(f"diagnostics={json_path}")
    if not args.no_plot:
        print(f"plot={plot_path}")
    print()
    print(summary.loc[summary.group_type.isin(["overall", "compat_quartile"]),
                      ["group_type", "group", "count_rows", "mean_compatibility",
                       "baseline_mae", "cfcompat_mae", "mean_gain",
                       "gain_ci95_low", "gain_ci95_high", "improved_rate"]].to_string(index=False))
    print()
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
