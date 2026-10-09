"""Validation-only CFCompat audit: relative gaps, transfer utility, intensity.

This audit DOES NOT retrain, select checkpoints, load Test data, or establish
causality. It deliberately combines predictions from three frozen models:
  evaluator/NoKD (ModDrop), Uniform KD (FixedKD), and CFCompat.
A separately trained clean Teacher is optional for independent KD-gap analysis.

The key falsifiable questions are:
  (1) Can near-equal raw counterfactual gaps map to different train-referenced
      empirical percentiles under different target conditions?
  (2) Does condition-wise calibration explain independent disagreement or
      measured KD benefit better than pooled calibration across conditions?
  (3) When Uniform KD and CFCompat both get polarity right, does CFCompat
      improve continuous sentiment magnitude (MAE)?

Within one target condition, empirical rank is monotonic in raw discrepancy;
it adds NO new information or ordering inside that condition. Across conditions,
it changes the reference scale. Correlation is not proof of causal transfer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ("LA", "LV", "L")
MODE_PAIRS = (("LA", "LV"), ("LA", "L"), ("LV", "L"))


def midpoint_compatibility(train, values):
    """C=1-train midpoint ECDF, matching the validation-analysis convention."""
    ref = np.sort(np.asarray(train, dtype=np.float64).reshape(-1))
    val = np.asarray(values, dtype=np.float64).reshape(-1)
    if ref.size == 0 or not np.isfinite(ref).all() or (ref < 0).any():
        raise ValueError("Train discrepancies must be finite and nonnegative.")
    if not np.isfinite(val).all() or (val < 0).any():
        raise ValueError("Validation discrepancies must be finite and nonnegative.")
    left = np.searchsorted(ref, val, side="left").astype(float)
    right = np.searchsorted(ref, val, side="right").astype(float)
    quantile = (left + .5 * (right - left)) / len(ref)
    quantile = np.clip(quantile, .5 / len(ref), 1 - .5 / len(ref))
    return 1.0 - quantile


def forbid_test_path(path):
    if path is None:
        return
    components = [p.lower() for p in Path(path).parts]
    name = Path(path).name.lower()
    if "test" in name or "test" in components:
        raise ValueError("Validation-only audit refuses a Test input/output path: " + str(path))


def require_cols(frame, required, path):
    absent = sorted(set(required) - set(frame.columns))
    if absent:
        raise ValueError("%s is missing columns: %s" % (path, absent))


def read_paired_predictions(path, required):
    forbid_test_path(path)
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    require_cols(frame, required, path)
    if frame.sample_index.isna().any() or frame.sample_index.duplicated().any():
        raise ValueError("%s has invalid/duplicate sample_index" % path)
    for key in ["sample_index", "label"] + [x for x in required if x.endswith("_pred")]:
        if not np.isfinite(pd.to_numeric(frame[key], errors="coerce")).all():
            raise ValueError("%s has NaN/Inf/non-numeric %s" % (path, key))
    return frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def validate_alignment(reference, other, description):
    if len(other) != len(reference):
        raise ValueError("%s count mismatches evaluator" % description)
    for col in ("sample_index", "sample_id"):
        if col in reference.columns and col in other.columns:
            if not np.array_equal(reference[col].astype(str), other[col].astype(str)):
                raise ValueError("%s is not paired by %s" % (description, col))
    if not np.allclose(reference.label.to_numpy(float),
                       other.label.to_numpy(float), rtol=0, atol=1e-8):
        raise ValueError("%s labels mismatch evaluator" % description)


def load_data(args):
    root = Path(args.result_root)
    seed = args.seed
    dataset = args.dataset
    evaluator = Path(args.evaluator_valid or
        root / "missing_baseline" / "moddrop_benchmark_multiseed_v1" /
        ("seed%d" % seed) / (dataset + "_best_valid_predictions.csv"))
    fixed = Path(args.uniform_valid or
        root / "analysis" / "compatibility_gain_v2" / dataset /
        ("seed%d" % seed) / ("%s_seed%d_fixedkd_valid_predictions.csv" % (dataset, seed)))
    ours = Path(args.cfcompat_valid or
        root / "missing_baseline" / "cf_compat_kd_v1" / "benchmark_multiseed" /
        ("seed%d" % seed) / (dataset + "_best_valid_predictions.csv"))
    train = Path(args.train_cache or
        root / "counterfactual_compatibility" / "cf_compat_v1_multiseed" /
        dataset / ("seed%d" % seed) / "train_counterfactual_compatibility.csv")
    teacher = Path(args.teacher_valid) if args.teacher_valid else None
    out = Path(args.output_dir or root / "analysis" / "relative_gap_transfer_v1" /
               dataset / ("seed%d" % seed))
    for path in (evaluator, fixed, ours, train, teacher, out):
        forbid_test_path(path)
    prediction_columns = ["sample_index", "label", "LAV_pred"] + [
        m + "_pred" for m in MODES
    ]
    ref = read_paired_predictions(evaluator, prediction_columns)
    uniform = read_paired_predictions(fixed, prediction_columns)
    compat = read_paired_predictions(ours, prediction_columns)
    validate_alignment(ref, uniform, "Uniform KD")
    validate_alignment(ref, compat, "CFCompat")
    if not train.is_file():
        raise FileNotFoundError(str(train))
    train_frame = pd.read_csv(train)
    require_cols(train_frame, ["sample_index"] + ["delta_" + m for m in MODES], train)
    if train_frame.sample_index.duplicated().any():
        raise ValueError("Train cache duplicates sample_index")
    for mode in MODES:
        d = train_frame["delta_" + mode].to_numpy(float)
        if not np.isfinite(d).all() or (d < 0).any():
            raise ValueError("Invalid train discrepancies for " + mode)
    teacher_frame = None
    if teacher is not None:
        teacher_frame = read_paired_predictions(
            teacher, ["sample_index", "label", "teacher_LAV_pred"]
        )
        validate_alignment(ref, teacher_frame, "Clean Teacher")
    return ref, uniform, compat, train_frame, teacher_frame, out, {
        "evaluator_valid": str(evaluator),
        "uniform_valid": str(fixed),
        "cfcompat_valid": str(ours),
        "train_cache": str(train),
        "teacher_valid": str(teacher) if teacher else None,
    }


def long_table(ref, fixed, ours, cache, teacher=None):
    train_pooled = np.concatenate([cache["delta_" + m].to_numpy(float) for m in MODES])
    y = ref.label.to_numpy(float)
    rows = []
    for mode in MODES:
        gap = np.abs(ref.LAV_pred.to_numpy(float) - ref[mode + "_pred"].to_numpy(float))
        c_mode = midpoint_compatibility(cache["delta_" + mode], gap)
        c_pooled = midpoint_compatibility(train_pooled, gap)
        no_kd_error = np.abs(ref[mode + "_pred"].to_numpy(float) - y)
        uni_error = np.abs(fixed[mode + "_pred"].to_numpy(float) - y)
        ours_error = np.abs(ours[mode + "_pred"].to_numpy(float) - y)
        item = pd.DataFrame({
            "sample_index": ref.sample_index.to_numpy(),
            "sample_id": (ref.sample_id.astype(str).to_numpy()
                          if "sample_id" in ref else ref.sample_index.astype(str).to_numpy()),
            "mode": mode, "label": y, "delta": gap,
            "compat_mode": c_mode, "compat_pooled": c_pooled,
            "no_kd_pred": ref[mode + "_pred"].to_numpy(float),
            "uniform_pred": fixed[mode + "_pred"].to_numpy(float),
            "cfcompat_pred": ours[mode + "_pred"].to_numpy(float),
            "no_kd_ae": no_kd_error, "uniform_ae": uni_error,
            "cfcompat_ae": ours_error,
            "uniform_gain": no_kd_error - uni_error,
            "compat_gain": uni_error - ours_error,
            "total_gain": no_kd_error - ours_error,
        })
        if teacher is not None:
            t = teacher.teacher_LAV_pred.to_numpy(float)
            item["teacher_student_gap"] = np.abs(t - fixed[mode + "_pred"].to_numpy(float))
            diff = np.abs(t - fixed[mode + "_pred"].to_numpy(float))
            item["teacher_student_smoothl1"] = np.where(diff < 1, .5 * diff**2, diff - .5)
        rows.append(item)
    result = pd.concat(rows, ignore_index=True)
    result.sort_values(["sample_index", "mode"], inplace=True, kind="mergesort")
    return result.reset_index(drop=True)


def nearest_same_sample_pairs(long, tolerance, min_rank_difference):
    """Within a single validation sample match different modes on raw delta."""
    by_sample = {int(idx): g.set_index("mode") for idx, g in long.groupby("sample_index")}
    matches = []
    for idx in sorted(by_sample):
        data = by_sample[idx]
        for m1, m2 in MODE_PAIRS:
            a, b = data.loc[m1], data.loc[m2]
            diff = abs(float(a.delta) - float(b.delta))
            rank_diff = abs(float(a.compat_mode) - float(b.compat_mode))
            if diff > tolerance or rank_diff < min_rank_difference:
                continue
            hi, lo = (a, b) if a.compat_mode >= b.compat_mode else (b, a)
            match = {
                "sample_index": idx, "sample_id": str(a.sample_id),
                "label": float(a.label),
                "high_compat_mode": str(hi.name),
                "low_compat_mode": str(lo.name),
                "high_delta": float(hi.delta), "low_delta": float(lo.delta),
                "abs_delta_difference": diff,
                "high_compat": float(hi.compat_mode),
                "low_compat": float(lo.compat_mode),
                "compat_separation": rank_diff,
                "high_uniform_gain": float(hi.uniform_gain),
                "low_uniform_gain": float(lo.uniform_gain),
                "high_compat_gain": float(hi.compat_gain),
                "low_compat_gain": float(lo.compat_gain),
            }
            if "teacher_student_gap" in long:
                match["high_teacher_student_gap"] = float(hi.teacher_student_gap)
                match["low_teacher_student_gap"] = float(lo.teacher_student_gap)
            matches.append(match)
    columns = [
        "sample_index", "sample_id", "label", "high_compat_mode", "low_compat_mode",
        "high_delta", "low_delta", "abs_delta_difference", "high_compat",
        "low_compat", "compat_separation", "high_uniform_gain", "low_uniform_gain",
        "high_compat_gain", "low_compat_gain"
    ] + (["high_teacher_student_gap", "low_teacher_student_gap"]
         if "teacher_student_gap" in long else [])
    return pd.DataFrame(matches, columns=columns).sort_values(
        ["compat_separation", "abs_delta_difference", "sample_index"],
        ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)


def cluster_bootstrap_mean(frame, col, n_boot, rng):
    if frame.empty:
        return [float("nan"), float("nan")]
    clusters = [
        group[col].to_numpy(float)
        for _, group in frame.groupby("sample_index", sort=True)
    ]
    stats = np.empty(n_boot)
    for j in range(n_boot):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([clusters[int(k)] for k in picks])
        stats[j] = float(np.mean(values))
    return list(np.quantile(stats, [.025, .975]))


def correlation_summary(frame, bootstrap_reps, seed):
    """Cluster-bootstrap differences in pooled rank correlation, not causal effects."""
    rng = np.random.default_rng(seed)
    outcomes = ["uniform_gain", "compat_gain"]
    if "teacher_student_gap" in frame:
        outcomes += ["teacher_student_gap", "teacher_student_smoothl1"]
    cluster_indices = [grp.index.to_numpy() for _, grp in frame.groupby("sample_index")]
    result = []
    for subset_name, local in [("all", frame)] + [
        (mode, frame.loc[frame["mode"] == mode]) for mode in MODES
    ]:
        for outcome in outcomes:
            s_mode = local["compat_mode"].corr(local[outcome], method="spearman")
            s_pool = local["compat_pooled"].corr(local[outcome], method="spearman")
            # Only the pooled rows require cluster-resampling across all three modes.
            ci_low, ci_high = float("nan"), float("nan")
            if subset_name == "all" and bootstrap_reps:
                differences = []
                for _ in range(bootstrap_reps):
                    picks = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
                    indices = np.concatenate([cluster_indices[int(k)] for k in picks])
                    sample = frame.iloc[indices]
                    r1 = sample.compat_mode.corr(sample[outcome], method="spearman")
                    r2 = sample.compat_pooled.corr(sample[outcome], method="spearman")
                    if np.isfinite(r1) and np.isfinite(r2):
                        differences.append(float(r1-r2))
                if differences:
                    ci_low, ci_high = np.quantile(differences, [.025,.975])
            result.append({
                "subset": subset_name, "outcome": outcome, "n": len(local),
                "spearman_modewise": float(s_mode),
                "spearman_pooled": float(s_pool),
                "modewise_minus_pooled": float(s_mode - s_pool),
                "difference_ci95_low": ci_low, "difference_ci95_high": ci_high,
            })
    return pd.DataFrame(result)


def intensity_bins(y):
    a = np.abs(np.asarray(y, dtype=float))
    return np.select(
        [a <= 1.0, a <= 2.0],
        ["mild_or_neutral", "moderate"], default="strong"
    )


def intensity_summary(frame, bootstrap_reps, seed):
    """No neutral samples; both methods must get binary polarity correct."""
    rows = frame.loc[frame.label != 0].copy()
    rows["both_polarity_correct"] = (
        (np.sign(rows.uniform_pred) == np.sign(rows.label))
        & (np.sign(rows.cfcompat_pred) == np.sign(rows.label))
    )
    rows["intensity_region"] = intensity_bins(rows.label)
    rows = rows.loc[rows.both_polarity_correct].copy()
    rows["error_reduction"] = rows.uniform_ae - rows.cfcompat_ae
    rng = np.random.default_rng(seed)
    stats = []
    for condition, cond in [("all", rows)] + [
        (mode, rows.loc[rows["mode"] == mode]) for mode in MODES
    ]:
        for region in ("all", "mild_or_neutral", "moderate", "strong"):
            local = cond if region == "all" else cond.loc[cond.intensity_region == region]
            if local.empty:
                continue
            low, high = cluster_bootstrap_mean(
                local, "error_reduction", bootstrap_reps, rng
            ) if bootstrap_reps else (float("nan"), float("nan"))
            stats.append({
                "condition": condition, "intensity_region": region,
                "n_view_pairs": len(local),
                "n_unique_samples": local.sample_index.nunique(),
                "uniform_MAE": float(local.uniform_ae.mean()),
                "cfcompat_MAE": float(local.cfcompat_ae.mean()),
                "MAE_reduction": float(local.error_reduction.mean()),
                "reduction_ci95_low": low, "reduction_ci95_high": high,
                "cfcompat_better_fraction": float((local.error_reduction > 0).mean()),
            })
    return pd.DataFrame(stats)


def gap_bin_summary(long, cache):
    """Train-only pooled quantiles define bins; Valid never sets the cutoffs."""
    pooled = np.concatenate([cache["delta_" + m].to_numpy(float) for m in MODES])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, 11)))
    # Bound the external Valid samples without changing training-derived cutoffs.
    edges[0] = -np.inf
    edges[-1] = np.inf
    copy = long.copy()
    copy["pooled_train_gap_decile"] = pd.cut(copy.delta, bins=edges,
                                            include_lowest=True, labels=False)
    summary = copy.groupby(["pooled_train_gap_decile", "mode"], observed=True).agg(
        n=("delta", "size"), median_delta=("delta", "median"),
        mean_mode_compat=("compat_mode", "mean"),
        mean_pooled_compat=("compat_pooled", "mean"),
        mean_uniform_gain=("uniform_gain", "mean"),
        mean_cfcompat_gain=("compat_gain", "mean"),
        mean_uniform_MAE=("uniform_ae", "mean"),
        mean_cfcompat_MAE=("cfcompat_ae", "mean"),
    ).reset_index()
    if "teacher_student_gap" in copy:
        g = copy.groupby(["pooled_train_gap_decile", "mode"],
                         observed=True)["teacher_student_gap"].mean().rename(
                             "mean_teacher_student_gap"
                         ).reset_index()
        summary = summary.merge(g, on=["pooled_train_gap_decile","mode"], validate="one_to_one")
    return summary


def plot_comparison(frame, cache, matches, png, pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LA": "#5279B8", "LV": "#459A86", "L": "#D4954A"}
    fig, axs = plt.subplots(1, 2, figsize=(10.8, 3.45))
    for mode in MODES:
        grid = np.sort(cache["delta_" + mode].to_numpy(float))
        y = (np.arange(len(grid)) + .5) / len(grid)
        axs[0].plot(grid, y, label=mode, color=colors[mode], lw=2)
    axs[0].set_xlabel("Raw counterfactual discrepancy")
    axs[0].set_ylabel("Empirical cumulative probability (Train)")
    axs[0].set_title("(a) Condition-specific discrepancy distributions")
    axs[0].legend(frameon=False)

    if not matches.empty:
        example = matches.iloc[0]
        for mode, delta, compat in [
            (example.high_compat_mode, example.high_delta, example.high_compat),
            (example.low_compat_mode, example.low_delta, example.low_compat),
        ]:
            axs[0].scatter([delta], [1 - compat], marker="o", s=65,
                           color=colors[mode], edgecolor="white", zorder=4)
        axs[1].barh(
            [0, 1], [example.high_compat, example.low_compat],
            color=[colors[example.high_compat_mode],
                   colors[example.low_compat_mode]], height=.42
        )
        axs[1].set_yticks([0,1])
        axs[1].set_yticklabels(
            [str(example.high_compat_mode) + " (higher)",
             str(example.low_compat_mode) + " (lower)"])
        axs[1].invert_yaxis()
        axs[1].set_xlim(0, 1)
        axs[1].set_xlabel("Condition-wise compatibility")
        axs[1].set_title("(b) Near-equal gap, same validation sample")
        axs[1].text(.02, .08,
                    "gaps: %.3f vs %.3f; sample %s" %
                    (example.high_delta, example.low_delta, example.sample_index),
                    transform=axs[1].transAxes, fontsize=8.8,
                    va="bottom")
    else:
        axs[1].axis("off")
        axs[1].text(.5, .5, "No matched pairs under fixed thresholds",
                    ha="center", va="center", transform=axs[1].transAxes)
    fig.tight_layout(pad=1.2)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--result-root", default="result")
    p.add_argument("--evaluator-valid")
    p.add_argument("--uniform-valid")
    p.add_argument("--cfcompat-valid")
    p.add_argument("--train-cache")
    p.add_argument("--teacher-valid", help="Optional clean Teacher validation predictions")
    p.add_argument("--output-dir")
    p.add_argument("--match-tolerance", type=float, default=None,
                   help="Maximum raw-gap difference; default = 5%% of train pooled IQR")
    p.add_argument("--min-rank-separation", type=float, default=.25)
    p.add_argument("--bootstrap-reps", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=20261009)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    if args.bootstrap_reps < 0 or (args.match_tolerance is not None and args.match_tolerance < 0):
        p.error("bootstrap-reps and match-tolerance must be nonnegative")
    if not (0 <= args.min_rank_separation <= 1):
        p.error("min-rank-separation must lie in [0,1]")
    return args


def main():
    args = parse_args()
    ref, fixed, ours, cache, teacher, out, sources = load_data(args)
    pooled_train = np.concatenate([cache["delta_" + m].to_numpy(float) for m in MODES])
    iqr = float(np.quantile(pooled_train,.75) - np.quantile(pooled_train,.25))
    tolerance = args.match_tolerance if args.match_tolerance is not None else .05 * iqr
    long = long_table(ref, fixed, ours, cache, teacher)
    matched = nearest_same_sample_pairs(long, tolerance, args.min_rank_separation)
    corrs = correlation_summary(long, args.bootstrap_reps, args.bootstrap_seed)
    intensity = intensity_summary(long, args.bootstrap_reps, args.bootstrap_seed+1)
    binned = gap_bin_summary(long, cache)

    out.mkdir(parents=True, exist_ok=True)
    stem = "%s_seed%d" % (args.dataset, args.seed)
    artifacts = {
        "per_sample": stem+"_valid_relative_gap_rows.csv",
        "same_sample_pairs": stem+"_same_sample_matched_gap_pairs.csv",
        "rank_correlation": stem+"_modewise_vs_pooled_rank_correlation.csv",
        "intensity": stem+"_polarity_correct_intensity.csv",
        "pooled_gap_deciles": stem+"_fixed_gap_deciles_by_mode.csv",
    }
    for key, value in [(long,artifacts["per_sample"]),
                       (matched,artifacts["same_sample_pairs"]),
                       (corrs,artifacts["rank_correlation"]),
                       (intensity,artifacts["intensity"]),
                       (binned,artifacts["pooled_gap_deciles"])]:
        key.to_csv(out / value, index=False)
    if not args.no_plot:
        plot_comparison(long, cache, matched,
                        out/(stem+"_relative_gap_motivation.png"),
                        out/(stem+"_relative_gap_motivation.pdf"))
    diagnostics = {
        "protocol": "validation-only; no training, checkpoints, or Test access",
        "seed": args.seed, "dataset": args.dataset,
        "source_paths": sources,
        "n_valid_samples": int(long.sample_index.nunique()),
        "n_view_rows": int(len(long)),
        "train_pooled_iqr": iqr,
        "match_tolerance": float(tolerance),
        "min_rank_separation": args.min_rank_separation,
        "n_matched_view_pairs": int(len(matched)),
        "n_distinct_matched_samples": int(matched.sample_index.nunique()) if len(matched) else 0,
        "matched_pair_example_is_descriptive": True,
        "independent_teacher_gap_available": teacher is not None,
        "bootstrap_resamples_by_sample_index": args.bootstrap_reps,
        "important_limits": [
            "Within each condition rank is monotone in raw gap, so no new information is added.",
            "Near-equal raw-gap pairs and pooled correlations are descriptive, not causal.",
            "Cross-condition gain comparisons confound the target input condition.",
            "Positive correlation or mean gain does not establish individual KD benefit.",
            "This cannot compare CFCompat against CMAD without paired CMAD predictions.",
        ],
    }
    (out/(stem+"_protocol_and_diagnostics.json")).write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Validation-only relative-discrepancy audit complete.")
    print("output_dir:", out)
    print("matched_pairs:", len(matched), "distinct_samples:", diagnostics["n_distinct_matched_samples"])
    print("raw_gap_tolerance:", round(tolerance,6))
    print("\nAggregate score associations (do not infer causality):")
    print(corrs.loc[corrs.subset=="all"].to_string(index=False))
    print("\nBoth-polarity-correct MAE analysis:")
    print(intensity.loc[intensity.condition=="all"].to_string(index=False))
    print("\nReturn the per-sample, matched-pairs, rank-correlation, intensity and diagnostics files.")


if __name__ == "__main__":
    main()
