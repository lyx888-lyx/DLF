"""Train-only analysis of raw discrepancy scales and mode-wise compatibility calibration.

This script uses only the cached training counterfactual predictions/deltas.

It answers two questions:
1) Are raw counterfactual discrepancies on LA/LV/L directly comparable?
2) What changes if one uses a single pooled/global calibration instead of the
   proposed mode-wise empirical-rank calibration?

Outputs:
- raw discrepancy summary by mode
- pooled/global vs mode-wise compatibility summary
- raw discrepancy ECDF figure
- mode-wise calibration mapping figure
- pooled/global calibration bias figure

No validation or test split is read.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("LA", "LV", "L")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze raw discrepancy and compatibility calibration.")
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--result-root", default="result")
    p.add_argument("--train-cache")
    p.add_argument("--output-dir")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def default_paths(args):
    root = Path(args.result_root)
    cache = Path(args.train_cache) if args.train_cache else (
        root / "counterfactual_compatibility" / "cf_compat_v1_multiseed"
        / args.dataset / f"seed{args.seed}" / "train_counterfactual_compatibility.csv"
    )
    output = Path(args.output_dir) if args.output_dir else (
        root / "analysis" / "compatibility_calibration_v1" / args.dataset / f"seed{args.seed}"
    )
    return cache, output


def midpoint_compatibility(reference, values):
    """Compatibility = 1 - empirical midpoint CDF under a reference vector."""
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    vals = np.asarray(values, dtype=np.float64)
    if len(ref) == 0 or not np.isfinite(ref).all() or not np.isfinite(vals).all():
        raise ValueError("Calibration requires finite, non-empty values.")
    if np.any(ref < 0) or np.any(vals < 0):
        raise ValueError("Discrepancies must be non-negative.")

    left = np.searchsorted(ref, vals, side="left").astype(np.float64)
    right = np.searchsorted(ref, vals, side="right").astype(np.float64)
    q = (left + 0.5 * (right - left)) / float(len(ref))
    lo = 0.5 / float(len(ref))
    hi = 1.0 - lo
    q = np.clip(q, lo, hi)
    return 1.0 - q


def distribution_summary(values):
    x = np.asarray(values, dtype=np.float64)
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "p95": float(q[5]),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def analyze(cache):
    raw_rows = []
    long_rows = []

    pooled = np.concatenate([cache[f"delta_{m}"].to_numpy(dtype=np.float64) for m in MODES])

    for mode in MODES:
        delta = cache[f"delta_{mode}"].to_numpy(dtype=np.float64)
        modewise = cache[f"compat_{mode}"].to_numpy(dtype=np.float64)
        recomputed = midpoint_compatibility(delta, delta)
        if not np.allclose(modewise, recomputed, atol=1e-12, rtol=0):
            raise ValueError(f"Cached compatibility for {mode} does not match mode-wise midpoint-rank calibration.")

        global_compat = midpoint_compatibility(pooled, delta)
        stats = distribution_summary(delta)
        raw_rows.append({
            "mode": mode,
            **{f"delta_{k}": v for k, v in stats.items()},
            "modewise_compat_mean": float(modewise.mean()),
            "modewise_compat_std": float(modewise.std()),
            "global_compat_mean": float(global_compat.mean()),
            "global_compat_std": float(global_compat.std()),
            "mean_global_minus_modewise": float((global_compat - modewise).mean()),
            "mean_abs_global_minus_modewise": float(np.abs(global_compat - modewise).mean()),
        })

        local = pd.DataFrame({
            "mode": mode,
            "sample_index": cache["sample_index"].to_numpy(),
            "delta": delta,
            "modewise_compatibility": modewise,
            "global_compatibility": global_compat,
            "global_minus_modewise": global_compat - modewise,
        })
        long_rows.append(local)

    raw_summary = pd.DataFrame(raw_rows)
    long_frame = pd.concat(long_rows, ignore_index=True)

    medians = raw_summary.set_index("mode")["delta_median"].to_dict()
    means = raw_summary.set_index("mode")["delta_mean"].to_dict()
    diagnostics = {
        "source": "train_only_counterfactual_cache",
        "modes": list(MODES),
        "pooled_delta_count": int(len(pooled)),
        "raw_mean_range": float(max(means.values()) - min(means.values())),
        "raw_median_range": float(max(medians.values()) - min(medians.values())),
        "raw_mean_max_over_min": float(max(means.values()) / max(min(means.values()), 1e-12)),
        "raw_median_max_over_min": float(max(medians.values()) / max(min(medians.values()), 1e-12)),
        "global_compat_mean_range": float(
            raw_summary["global_compat_mean"].max() - raw_summary["global_compat_mean"].min()
        ),
        "modewise_compat_mean_range": float(
            raw_summary["modewise_compat_mean"].max() - raw_summary["modewise_compat_mean"].min()
        ),
        "interpretation_note":
            "Mode-wise compatibility means are approximately 0.5 by construction; the useful diagnostic is whether raw delta scales and pooled/global calibration differ across modes.",
    }
    return raw_summary, long_frame, diagnostics


def make_raw_ecdf(long_frame, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for mode in MODES:
        x = np.sort(long_frame.loc[long_frame["mode"] == mode, "delta"].to_numpy(float))
        y = np.arange(1, len(x) + 1, dtype=np.float64) / float(len(x))
        ax.plot(x, y, linewidth=2.0, label=mode)

    ax.set_xlabel("Raw counterfactual discrepancy")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Raw Discrepancy across Target Conditions")
    ax.grid(axis="both", alpha=0.25)
    ax.legend(title="Condition", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_calibration_mapping(long_frame, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for mode in MODES:
        local = long_frame.loc[long_frame["mode"] == mode].sort_values("delta", kind="mergesort")
        # Plot quantile anchors for a calm paper figure.
        qs = np.linspace(0.0, 1.0, 101)
        indices = np.clip((qs * (len(local) - 1)).round().astype(int), 0, len(local) - 1)
        anchors = local.iloc[indices]
        ax.plot(
            anchors["delta"].to_numpy(float),
            anchors["modewise_compatibility"].to_numpy(float),
            linewidth=2.0,
            label=mode,
        )

    ax.set_xlabel("Raw counterfactual discrepancy")
    ax.set_ylabel("Calibrated compatibility")
    ax.set_title("Condition-wise Compatibility Calibration")
    ax.grid(axis="both", alpha=0.25)
    ax.legend(title="Condition", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_global_bias_plot(raw_summary, output_path):
    import matplotlib.pyplot as plt

    x = np.arange(len(MODES))
    values = [
        float(raw_summary.loc[raw_summary["mode"] == mode, "global_compat_mean"].iloc[0])
        for mode in MODES
    ]

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(x, values)
    ax.axhline(0.5, linewidth=1.2, linestyle="--", label="Mode-wise center (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODES)
    ax.set_xlabel("Target condition")
    ax.set_ylabel("Mean compatibility")
    ax.set_title("Bias from Pooled Global Calibration")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cache_path, output_dir = default_paths(args)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    cache = pd.read_csv(cache_path)
    required = ["sample_index"]
    for mode in MODES:
        required.extend([f"delta_{mode}", f"compat_{mode}"])
    missing = [c for c in required if c not in cache.columns]
    if missing:
        raise ValueError(f"Cache is missing columns: {missing}")

    raw_summary, long_frame, diagnostics = analyze(cache)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.dataset}_seed{args.seed}_raw_discrepancy_calibration_summary.csv"
    samples_path = output_dir / f"{args.dataset}_seed{args.seed}_raw_discrepancy_calibration_samples.csv"
    diagnostics_path = output_dir / f"{args.dataset}_seed{args.seed}_raw_discrepancy_calibration_diagnostics.json"
    raw_ecdf_path = output_dir / f"{args.dataset}_seed{args.seed}_raw_discrepancy_ecdf.png"
    mapping_path = output_dir / f"{args.dataset}_seed{args.seed}_modewise_calibration_mapping.png"
    bias_path = output_dir / f"{args.dataset}_seed{args.seed}_global_calibration_bias.png"

    raw_summary.to_csv(summary_path, index=False)
    long_frame.to_csv(samples_path, index=False)
    diagnostics["train_cache"] = str(cache_path)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.no_plot:
        make_raw_ecdf(long_frame, raw_ecdf_path)
        make_calibration_mapping(long_frame, mapping_path)
        make_global_bias_plot(raw_summary, bias_path)

    cols = [
        "mode", "delta_mean", "delta_std", "delta_median", "delta_p25", "delta_p75", "delta_p90",
        "global_compat_mean", "modewise_compat_mean", "mean_abs_global_minus_modewise"
    ]
    print("Train-only raw discrepancy / calibration analysis complete.")
    print(f"summary={summary_path}")
    print(f"diagnostics={diagnostics_path}")
    if not args.no_plot:
        print(f"raw_ecdf={raw_ecdf_path}")
        print(f"calibration_mapping={mapping_path}")
        print(f"global_bias={bias_path}")
    print()
    print(raw_summary[cols].to_string(index=False))
    print()
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
