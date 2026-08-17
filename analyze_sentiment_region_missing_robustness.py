"""Post-hoc sentiment-region and missing-modality robustness analysis.

This script is intentionally offline. It reads already-frozen prediction CSVs,
never constructs a dataset loader, never runs a model forward pass, never
selects checkpoints, and never searches ensemble/blend weights.

Primary questions:
1) Where does the full-modality (LAV) gain come from in sentiment space?
2) Does Ours degrade more gracefully than the baseline as modalities disappear?

See SENTIMENT_REGION_MISSING_ROBUSTNESS_V1.md for the frozen analysis protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from trains.singleTask.missing_utils import regression_metrics


MODES: Tuple[str, ...] = ("LAV", "LA", "LV", "L")
PRED_COLUMNS: Dict[str, str] = {mode: f"{mode}_pred" for mode in MODES}

# Frozen primary bins. Reuse these unchanged on MOSI and MOSEI.
INTENSITY_ORDER: Tuple[str, ...] = (
    "near_neutral",
    "mild",
    "moderate",
    "extreme",
)

SIGNED_ORDER: Tuple[str, ...] = (
    "strong_negative",
    "moderate_negative",
    "mild_negative",
    "near_neutral",
    "mild_positive",
    "moderate_positive",
    "strong_positive",
)

METRIC_KEYS: Tuple[str, ...] = (
    "acc_7",
    "acc_5",
    "acc_2",
    "F1_score",
    "Corr",
    "MAE",
)

HIGHER_IS_BETTER = {"acc_7", "acc_5", "acc_2", "F1_score", "Corr"}
LOWER_IS_BETTER = {"MAE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline sentiment-region and missing-modality robustness analysis"
    )
    parser.add_argument("--dataset", choices=("mosi", "mosei"), required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--baseline-name", default="DLF")

    ours = parser.add_mutually_exclusive_group(required=True)
    ours.add_argument("--ours-predictions")
    ours.add_argument(
        "--ours-left-predictions",
        help="First frozen component used for offline fixed composition",
    )
    parser.add_argument(
        "--ours-right-predictions",
        help="Second frozen component; required with --ours-left-predictions",
    )
    parser.add_argument(
        "--blend-lambda",
        type=float,
        help="Frozen lambda for Ours=(1-lambda)*left + lambda*right; no search is performed",
    )
    parser.add_argument("--ours-name", default="Ours")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-test-analysis",
        action="store_true",
        help="Required for conscious post-hoc analysis of already-generated Test predictions",
    )
    parser.add_argument(
        "--rolling-fraction",
        type=float,
        default=0.08,
        help="Fraction of samples used by the descriptive rolling error-gain curve",
    )
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test_analysis:
        parser.error(
            "Test analysis is post-hoc and must be acknowledged with --allow-test-analysis."
        )
    if args.ours_left_predictions is not None:
        if args.ours_right_predictions is None or args.blend_lambda is None:
            parser.error(
                "--ours-left-predictions requires --ours-right-predictions and --blend-lambda."
            )
        if not (0.0 <= float(args.blend_lambda) <= 1.0):
            parser.error("--blend-lambda must lie in [0, 1].")
    elif args.ours_right_predictions is not None or args.blend_lambda is not None:
        parser.error(
            "--ours-right-predictions/--blend-lambda are only valid with --ours-left-predictions."
        )
    if not (0.01 <= float(args.rolling_fraction) <= 0.5):
        parser.error("--rolling-fraction must lie in [0.01, 0.5].")
    return args


def intensity_region(label: float) -> str:
    value = abs(float(label))
    if value <= 0.5:
        return "near_neutral"
    if value <= 1.5:
        return "mild"
    if value <= 2.5:
        return "moderate"
    return "extreme"


def signed_region(label: float) -> str:
    y = float(label)
    if y < -2.5:
        return "strong_negative"
    if y < -1.5:
        return "moderate_negative"
    if y < -0.5:
        return "mild_negative"
    if y <= 0.5:
        return "near_neutral"
    if y <= 1.5:
        return "mild_positive"
    if y <= 2.5:
        return "moderate_positive"
    return "strong_positive"


def available_modes(frame: pd.DataFrame) -> Tuple[str, ...]:
    return tuple(mode for mode in MODES if PRED_COLUMNS[mode] in frame.columns)


def load_prediction_frame(path: str, label: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"{label} prediction file not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"sample_index", "label", "LAV_pred"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")
    if frame.sample_index.duplicated().any():
        raise ValueError(f"{label} contains duplicate sample_index values.")
    numeric_columns = ["sample_index", "label"] + [
        PRED_COLUMNS[m] for m in available_modes(frame)
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError(f"{label} contains NaN/Inf in identity/prediction columns.")
    local = frame.copy()
    local["sample_index"] = local.sample_index.astype(np.int64)
    if "sample_id" not in local.columns:
        local["sample_id"] = local.sample_index.astype(str)
    return local.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def bind_frames(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    if not np.array_equal(
        left.sample_index.to_numpy(np.int64), right.sample_index.to_numpy(np.int64)
    ):
        raise RuntimeError(f"sample_index differs between {left_name} and {right_name}.")
    if len(left) != len(right):
        raise RuntimeError(f"sample count differs between {left_name} and {right_name}.")
    label_diff = np.max(
        np.abs(left.label.to_numpy(np.float64) - right.label.to_numpy(np.float64))
    )
    if label_diff > 1e-6:
        raise RuntimeError(
            f"label binding differs between {left_name} and {right_name}; max diff={label_diff}."
        )
    if "sample_id" in left.columns and "sample_id" in right.columns:
        if not np.array_equal(
            left.sample_id.astype(str).to_numpy(), right.sample_id.astype(str).to_numpy()
        ):
            raise RuntimeError(f"sample_id differs between {left_name} and {right_name}.")


def compose_fixed_blend(left: pd.DataFrame, right: pd.DataFrame, lam: float) -> pd.DataFrame:
    bind_frames(left, right, "ours-left", "ours-right")
    shared_modes = tuple(
        mode for mode in MODES if PRED_COLUMNS[mode] in left and PRED_COLUMNS[mode] in right
    )
    if "LAV" not in shared_modes:
        raise RuntimeError("Both Ours components must provide LAV_pred.")
    result = left[["sample_index", "sample_id", "label"]].copy()
    for mode in shared_modes:
        column = PRED_COLUMNS[mode]
        result[column] = (
            (1.0 - float(lam)) * left[column].to_numpy(np.float64)
            + float(lam) * right[column].to_numpy(np.float64)
        )
    return result


def exact_metrics(prediction: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    if len(label) == 0:
        return {key: float("nan") for key in METRIC_KEYS}
    pred_tensor = torch.as_tensor(prediction, dtype=torch.float32).view(-1, 1)
    label_tensor = torch.as_tensor(label, dtype=torch.float32).view(-1, 1)
    values = regression_metrics(pred_tensor, label_tensor)
    return {key: float(values[key]) for key in METRIC_KEYS}


def descriptive_metrics(prediction: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    if len(label) == 0:
        return {
            "MAE": float("nan"),
            "Bias": float("nan"),
            "RMSE": float("nan"),
            "Within0p5": float("nan"),
            "Within1p0": float("nan"),
        }
    error = prediction - label
    return {
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "Within0p5": float(np.mean(np.abs(error) <= 0.5)),
        "Within1p0": float(np.mean(np.abs(error) <= 1.0)),
    }


def overall_mode_metrics(
    baseline: pd.DataFrame,
    ours: pd.DataFrame,
    modes: Sequence[str],
    baseline_name: str,
    ours_name: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    labels = baseline.label.to_numpy(np.float64)
    for method_name, frame in ((baseline_name, baseline), (ours_name, ours)):
        by_mode: Dict[str, Dict[str, float]] = {}
        for mode in modes:
            by_mode[mode] = exact_metrics(
                frame[PRED_COLUMNS[mode]].to_numpy(np.float64), labels
            )
        lav = by_mode["LAV"]
        for mode in modes:
            metric = by_mode[mode]
            row: Dict[str, object] = {"Method": method_name, "Mode": mode, **metric}
            row["MAE_degradation_vs_LAV"] = metric["MAE"] - lav["MAE"]
            for key in HIGHER_IS_BETTER:
                row[f"{key}_drop_vs_LAV"] = lav[key] - metric[key]
            rows.append(row)
    return pd.DataFrame(rows)


def bin_metric_table(
    frame: pd.DataFrame,
    modes: Sequence[str],
    method_name: str,
    region_kind: str,
) -> pd.DataFrame:
    labels = frame.label.to_numpy(np.float64)
    if region_kind == "intensity":
        regions = np.asarray([intensity_region(y) for y in labels], dtype=object)
        order = INTENSITY_ORDER
    elif region_kind == "signed":
        regions = np.asarray([signed_region(y) for y in labels], dtype=object)
        order = SIGNED_ORDER
    else:
        raise ValueError(region_kind)

    rows: List[Dict[str, object]] = []
    total = len(frame)
    for mode in modes:
        predictions = frame[PRED_COLUMNS[mode]].to_numpy(np.float64)
        for region in order:
            mask = regions == region
            count = int(mask.sum())
            pred = predictions[mask]
            lab = labels[mask]
            exact = exact_metrics(pred, lab)
            desc = descriptive_metrics(pred, lab)
            row: Dict[str, object] = {
                "Method": method_name,
                "Mode": mode,
                "Region": region,
                "N": count,
                "Share": float(count / total) if total else float("nan"),
                **exact,
            }
            # Keep the same MAE value but add interpretable diagnostics.
            row.update({k: v for k, v in desc.items() if k != "MAE"})
            rows.append(row)
    return pd.DataFrame(rows)


def sample_error_gain(
    baseline: pd.DataFrame,
    ours: pd.DataFrame,
    modes: Sequence[str],
) -> pd.DataFrame:
    labels = baseline.label.to_numpy(np.float64)
    rows: List[pd.DataFrame] = []
    identity = baseline[["sample_index", "sample_id", "label"]].copy()
    identity["IntensityRegion"] = [intensity_region(y) for y in labels]
    identity["SignedRegion"] = [signed_region(y) for y in labels]
    for mode in modes:
        local = identity.copy()
        base_pred = baseline[PRED_COLUMNS[mode]].to_numpy(np.float64)
        ours_pred = ours[PRED_COLUMNS[mode]].to_numpy(np.float64)
        base_abs = np.abs(base_pred - labels)
        ours_abs = np.abs(ours_pred - labels)
        local["Mode"] = mode
        local["BaselinePrediction"] = base_pred
        local["OursPrediction"] = ours_pred
        local["BaselineAbsError"] = base_abs
        local["OursAbsError"] = ours_abs
        local["ErrorGain"] = base_abs - ours_abs
        local["BaselineSignedError"] = base_pred - labels
        local["OursSignedError"] = ours_pred - labels
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


def intensity_gain_table(sample_gain: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for mode in MODES:
        mode_frame = sample_gain.loc[sample_gain.Mode.eq(mode)]
        if mode_frame.empty:
            continue
        for region in INTENSITY_ORDER:
            local = mode_frame.loc[mode_frame.IntensityRegion.eq(region)]
            if local.empty:
                continue
            gain = local.ErrorGain.to_numpy(np.float64)
            tol = 1e-12
            rows.append(
                {
                    "Mode": mode,
                    "Region": region,
                    "N": int(len(local)),
                    "Share": float(len(local) / len(mode_frame)),
                    "BaselineMAE": float(local.BaselineAbsError.mean()),
                    "OursMAE": float(local.OursAbsError.mean()),
                    "MAEGain": float(gain.mean()),
                    "MedianErrorGain": float(np.median(gain)),
                    "WinRate": float(np.mean(gain > tol)),
                    "TieRate": float(np.mean(np.abs(gain) <= tol)),
                    "LossRate": float(np.mean(gain < -tol)),
                }
            )
    return pd.DataFrame(rows)


def missing_level_table(overall: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method, local in overall.groupby("Method", sort=False):
        local = local.set_index("Mode")
        levels: List[Tuple[str, Sequence[str]]] = [("LAV", ("LAV",))]
        one_missing = tuple(mode for mode in ("LA", "LV") if mode in local.index)
        if one_missing:
            levels.append(("OneMissingMean", one_missing))
        if "L" in local.index:
            levels.append(("L", ("L",)))
        for level, modes in levels:
            row: Dict[str, object] = {
                "Method": method,
                "MissingLevel": level,
                "ContributingModes": ",".join(modes),
            }
            for key in METRIC_KEYS:
                row[key] = float(np.mean([float(local.loc[m, key]) for m in modes]))
            rows.append(row)
    result = pd.DataFrame(rows)
    degradation: List[Dict[str, object]] = []
    for method, local in result.groupby("Method", sort=False):
        lav = local.loc[local.MissingLevel.eq("LAV")].iloc[0]
        for _, row in local.iterrows():
            out = dict(row)
            out["MAE_degradation_vs_LAV"] = float(row.MAE - lav.MAE)
            out["MAE_relative_degradation_vs_LAV"] = (
                float((row.MAE - lav.MAE) / lav.MAE) if float(lav.MAE) != 0 else float("nan")
            )
            out["Corr_drop_vs_LAV"] = float(lav.Corr - row.Corr)
            out["acc_7_drop_vs_LAV"] = float(lav.acc_7 - row.acc_7)
            out["acc_5_drop_vs_LAV"] = float(lav.acc_5 - row.acc_5)
            out["acc_2_drop_vs_LAV"] = float(lav.acc_2 - row.acc_2)
            out["F1_score_drop_vs_LAV"] = float(lav.F1_score - row.F1_score)
            degradation.append(out)
    return pd.DataFrame(degradation)


def save_label_distribution(frame: pd.DataFrame, out: Path) -> None:
    labels = frame.label.to_numpy(np.float64)
    counts = pd.Series([intensity_region(y) for y in labels]).value_counts()
    values = np.asarray([counts.get(region, 0) for region in INTENSITY_ORDER], dtype=float)
    shares = values / max(values.sum(), 1.0)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(INTENSITY_ORDER, shares)
    ax.set_ylabel("Sample share")
    ax.set_xlabel("Ground-truth sentiment intensity")
    ax.set_title("Label distribution by sentiment intensity")
    ax.set_ylim(0.0, max(0.05, float(shares.max()) * 1.2))
    for bar, share in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{share:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_lav_mae_by_intensity(
    intensity_metrics: pd.DataFrame,
    baseline_name: str,
    ours_name: str,
    out: Path,
) -> None:
    local = intensity_metrics.loc[intensity_metrics.Mode.eq("LAV")]
    x = np.arange(len(INTENSITY_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for offset, method in ((-width / 2, baseline_name), (width / 2, ours_name)):
        values = []
        method_frame = local.loc[local.Method.eq(method)].set_index("Region")
        for region in INTENSITY_ORDER:
            values.append(float(method_frame.loc[region, "MAE"]) if region in method_frame.index else np.nan)
        ax.bar(x + offset, values, width, label=method)
    ax.set_xticks(x, INTENSITY_ORDER)
    ax.set_ylabel("MAE (lower is better)")
    ax.set_xlabel("Ground-truth sentiment intensity")
    ax.set_title("Full-modality error by sentiment region")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_error_gain_curve(sample_gain: pd.DataFrame, rolling_fraction: float, out: Path) -> None:
    local = sample_gain.loc[sample_gain.Mode.eq("LAV")].sort_values("label", kind="mergesort")
    if local.empty:
        return
    n = len(local)
    window = max(11, int(round(n * rolling_fraction)))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else max(1, n - 1))
    smooth = local.ErrorGain.rolling(window=window, center=True, min_periods=max(3, window // 3)).mean()

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.scatter(local.label, local.ErrorGain, s=9, alpha=0.18, label="sample gain")
    ax.plot(local.label, smooth, linewidth=2.2, label=f"rolling mean (window={window})")
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlim(-3.1, 3.1)
    ax.set_xlabel("Ground-truth sentiment y")
    ax.set_ylabel("|e_baseline| - |e_ours|  (positive = Ours better)")
    ax.set_title("Where does the full-modality gain come from?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_missing_degradation(missing: pd.DataFrame, out: Path) -> None:
    level_order = [level for level in ("LAV", "OneMissingMean", "L") if level in set(missing.MissingLevel)]
    if len(level_order) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, local in missing.groupby("Method", sort=False):
        local = local.set_index("MissingLevel")
        values = [float(local.loc[level, "MAE"]) for level in level_order]
        ax.plot(level_order, values, marker="o", linewidth=2.2, label=method)
    ax.set_ylabel("MAE (lower is better)")
    ax.set_xlabel("Available modality condition")
    ax.set_title("Missing-modality performance degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_gain_heatmap(gain: pd.DataFrame, modes: Sequence[str], out: Path) -> None:
    matrix = np.full((len(INTENSITY_ORDER), len(modes)), np.nan, dtype=float)
    for i, region in enumerate(INTENSITY_ORDER):
        for j, mode in enumerate(modes):
            local = gain.loc[gain.Mode.eq(mode) & gain.Region.eq(region)]
            if len(local) == 1:
                matrix[i, j] = float(local.iloc[0].MAEGain)
    if np.isnan(matrix).all():
        return
    vmax = float(np.nanmax(np.abs(matrix)))
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(modes)), modes)
    ax.set_yticks(np.arange(len(INTENSITY_ORDER)), INTENSITY_ORDER)
    ax.set_xlabel("Modality condition")
    ax.set_ylabel("Ground-truth sentiment intensity")
    ax.set_title("MAE gain of Ours over baseline (positive = better)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Baseline MAE - Ours MAE")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    dataset: str,
    split: str,
    baseline_name: str,
    ours_name: str,
    overall: pd.DataFrame,
    intensity_gain: pd.DataFrame,
    missing: pd.DataFrame,
    modes: Sequence[str],
    composition: Optional[Dict[str, object]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "analysis_type": "offline_posthoc_prediction_analysis",
        "frozen_primary_bins": {
            "near_neutral": "|y| <= 0.5",
            "mild": "0.5 < |y| <= 1.5",
            "moderate": "1.5 < |y| <= 2.5",
            "extreme": "|y| > 2.5",
        },
        "available_modes": list(modes),
        "baseline_name": baseline_name,
        "ours_name": ours_name,
        "composition": composition,
    }

    lav_gain = intensity_gain.loc[intensity_gain.Mode.eq("LAV")].copy()
    if not lav_gain.empty:
        best = lav_gain.sort_values(["MAEGain", "N"], ascending=[False, False], kind="mergesort").iloc[0]
        summary["largest_mean_LAV_MAE_gain_region"] = {
            "region": str(best.Region),
            "N": int(best.N),
            "share": float(best.Share),
            "baseline_MAE": float(best.BaselineMAE),
            "ours_MAE": float(best.OursMAE),
            "MAE_gain": float(best.MAEGain),
            "win_rate": float(best.WinRate),
        }
        summary["LAV_intensity_regions"] = [
            {
                "region": str(row.Region),
                "N": int(row.N),
                "share": float(row.Share),
                "baseline_MAE": float(row.BaselineMAE),
                "ours_MAE": float(row.OursMAE),
                "MAE_gain": float(row.MAEGain),
                "win_rate": float(row.WinRate),
            }
            for _, row in lav_gain.iterrows()
        ]

    summary["overall_mode_metrics"] = overall.to_dict(orient="records")
    if not missing.empty:
        summary["missing_level_metrics"] = missing.to_dict(orient="records")
    return summary


def write_summary_markdown(summary: Dict[str, object], path: Path) -> None:
    lines: List[str] = []
    lines.append(f"# {summary['dataset'].upper()} {summary['split']} analysis summary")
    lines.append("")
    lines.append("This is an offline post-hoc analysis of already-generated predictions.")
    lines.append("")
    largest = summary.get("largest_mean_LAV_MAE_gain_region")
    if largest:
        lines.append("## Full-modality gain localization")
        lines.append("")
        lines.append(
            "Largest mean LAV MAE gain occurs in **{region}**: "
            "N={N}, share={share:.1%}, baseline MAE={baseline_MAE:.4f}, "
            "Ours MAE={ours_MAE:.4f}, gain={MAE_gain:+.4f}, win rate={win_rate:.1%}.".format(**largest)
        )
        lines.append("")
        lines.append("This is a descriptive result; interpret it together with all four bins and the error-gain curve.")
        lines.append("")
    if "missing_level_metrics" in summary:
        lines.append("## Missing-modality degradation")
        lines.append("")
        lines.append("Inspect `missing_level_metrics.csv` and `fig_missing_degradation_mae.png` for LAV -> one-missing -> L degradation.")
        lines.append("")
    lines.append("## Cross-dataset rule")
    lines.append("")
    lines.append("Reuse the same frozen bins on MOSEI; do not redefine regions after seeing MOSEI results.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cli = parse_args()
    baseline = load_prediction_frame(cli.baseline_predictions, "baseline")

    composition: Optional[Dict[str, object]] = None
    if cli.ours_predictions is not None:
        ours = load_prediction_frame(cli.ours_predictions, "ours")
    else:
        left = load_prediction_frame(cli.ours_left_predictions, "ours-left")
        right = load_prediction_frame(cli.ours_right_predictions, "ours-right")
        ours = compose_fixed_blend(left, right, cli.blend_lambda)
        composition = {
            "type": "fixed_two_component_blend",
            "formula": "(1-lambda)*left + lambda*right",
            "lambda": float(cli.blend_lambda),
            "left": str(Path(cli.ours_left_predictions)),
            "right": str(Path(cli.ours_right_predictions)),
            "weight_search_performed": False,
        }

    bind_frames(baseline, ours, cli.baseline_name, cli.ours_name)
    modes = tuple(
        mode
        for mode in MODES
        if PRED_COLUMNS[mode] in baseline.columns and PRED_COLUMNS[mode] in ours.columns
    )
    if "LAV" not in modes:
        raise RuntimeError("Both methods must provide LAV_pred.")

    out = Path(cli.output_dir) if cli.output_dir else (
        Path("result")
        / "analysis"
        / "sentiment_region_missing_robustness_v1"
        / cli.dataset
        / cli.split
    )
    out.mkdir(parents=True, exist_ok=True)

    overall = overall_mode_metrics(
        baseline, ours, modes, cli.baseline_name, cli.ours_name
    )
    baseline_intensity = bin_metric_table(baseline, modes, cli.baseline_name, "intensity")
    ours_intensity = bin_metric_table(ours, modes, cli.ours_name, "intensity")
    intensity_metrics = pd.concat([baseline_intensity, ours_intensity], ignore_index=True)

    baseline_signed = bin_metric_table(baseline, modes, cli.baseline_name, "signed")
    ours_signed = bin_metric_table(ours, modes, cli.ours_name, "signed")
    signed_metrics = pd.concat([baseline_signed, ours_signed], ignore_index=True)

    sample_gain = sample_error_gain(baseline, ours, modes)
    gain = intensity_gain_table(sample_gain)
    missing = missing_level_table(overall)

    overall.to_csv(out / "overall_mode_metrics.csv", index=False)
    intensity_metrics.to_csv(out / "intensity_bin_metrics.csv", index=False)
    gain.to_csv(out / "intensity_bin_gain.csv", index=False)
    signed_metrics.to_csv(out / "signed_bin_metrics.csv", index=False)
    sample_gain.to_csv(out / "sample_error_gain.csv", index=False)
    missing.to_csv(out / "missing_level_metrics.csv", index=False)
    missing.to_csv(out / "robustness_degradation.csv", index=False)

    save_label_distribution(baseline, out / "fig_label_distribution.png")
    save_lav_mae_by_intensity(
        intensity_metrics, cli.baseline_name, cli.ours_name, out / "fig_lav_mae_by_intensity.png"
    )
    save_error_gain_curve(sample_gain, cli.rolling_fraction, out / "fig_error_gain_vs_label.png")
    save_missing_degradation(missing, out / "fig_missing_degradation_mae.png")
    save_gain_heatmap(gain, modes, out / "fig_mae_gain_heatmap.png")

    summary = build_summary(
        cli.dataset,
        cli.split,
        cli.baseline_name,
        cli.ours_name,
        overall,
        gain,
        missing,
        modes,
        composition,
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(summary, out / "summary.md")

    print("Sentiment-region / missing-modality analysis complete")
    print("dataset:", cli.dataset)
    print("split:", cli.split)
    print("available modes:", ", ".join(modes))
    print("output:", out.resolve())
    if cli.split == "test":
        print("NOTE: this is acknowledged post-hoc Test analysis; no model forward was run.")
    largest = summary.get("largest_mean_LAV_MAE_gain_region")
    if largest:
        print(
            "largest mean LAV MAE gain region: {} (share={:.1%}, gain={:+.6f})".format(
                largest["region"], largest["share"], largest["MAE_gain"]
            )
        )


if __name__ == "__main__":
    main()
