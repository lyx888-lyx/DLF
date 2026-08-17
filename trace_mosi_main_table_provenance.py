"""Trace the provenance of the MOSI main-table Ours row.

This is a read-only post-hoc diagnostic. It complements the aggregate JSON/CSV
signature locators with two additional searches:

1) scan historical text/log/markdown outputs for Corr~.802 and MAE~.693 appearing
   in the same local context;
2) find sample-level prediction CSVs, recompute MOSI metrics from labels and
   prediction columns, and rank the reconstructed rows by Corr/MAE distance.

Nothing is trained, selected, calibrated, or modified.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

EXPECTED_MOSI_TEST_N = 686
TEXT_SUFFIXES = {".log", ".txt", ".md", ".out"}
LABEL_ALIASES = (
    "label", "labels", "target", "targets", "truth", "ground_truth", "gt", "y_true", "y",
)
GENERATED_TOKENS = (
    "posthoc_analysis",
    "main_table_metric_signature_matches.json",
    "corr_mae_signature_matches.json",
    "mosi_main_table_provenance_trace.json",
    "sentiment_artifact_discovery_mosi.json",
)


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def is_generated(path: Path) -> bool:
    lowered = str(path).replace("\\", "/").lower()
    return any(token.lower() in lowered for token in GENERATED_TOKENS)


def rounded_match(value: float, target: float, decimals: int = 3) -> bool:
    return round(float(value), decimals) == round(float(target), decimals)


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[mask], target[mask]
    if prediction.size == 0:
        raise ValueError("empty finite prediction/target pair")

    clipped_7_prediction = np.clip(prediction, -3.0, 3.0)
    clipped_7_target = np.clip(target, -3.0, 3.0)
    clipped_5_prediction = np.clip(prediction, -2.0, 2.0)
    clipped_5_target = np.clip(target, -2.0, 2.0)
    nonzero = target != 0
    if np.any(nonzero):
        binary_prediction = prediction[nonzero] > 0
        binary_target = target[nonzero] > 0
        acc2 = float(np.mean(binary_prediction == binary_target))
        f1 = float(f1_score(binary_target, binary_prediction, average="weighted", zero_division=0))
    else:
        acc2 = 0.0
        f1 = 0.0
    if prediction.size < 2 or np.std(prediction) == 0 or np.std(target) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "acc7": float(np.mean(np.round(clipped_7_prediction) == np.round(clipped_7_target))),
        "acc5": float(np.mean(np.round(clipped_5_prediction) == np.round(clipped_5_target))),
        "acc2": acc2,
        "f1": f1,
        "corr": corr,
        "mae": float(np.mean(np.abs(prediction - target))),
        "n": int(prediction.size),
    }


def find_label_column(frame: pd.DataFrame) -> Optional[str]:
    normalized = {norm(c): c for c in frame.columns}
    for alias in LABEL_ALIASES:
        key = norm(alias)
        if key in normalized:
            return normalized[key]
    return None


def candidate_prediction_columns(frame: pd.DataFrame, label_col: str) -> List[str]:
    result = []
    for column in frame.columns:
        if column == label_col:
            continue
        key = norm(column)
        if key in {"sampleindex", "index", "seed", "epoch", "fold", "id", "sampleid"}:
            continue
        if any(token in key for token in ("pred", "prediction", "outputlogit", "logit")):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().mean() >= 0.95:
                result.append(column)
    # Prefer complete-modality columns first, then generic predictions.
    def rank(column: str):
        k = norm(column)
        if k in {"lavpred", "lavprediction", "lavoutputlogit"} or "lav" in k:
            return (0, column)
        if k in {"prediction", "pred", "ypred", "outputlogit"}:
            return (1, column)
        return (2, column)
    return sorted(set(result), key=rank)


def infer_split(frame: pd.DataFrame, path: Path) -> str:
    for name in ("Split", "split"):
        if name in frame.columns:
            values = frame[name].dropna().astype(str).str.lower().unique().tolist()
            if len(values) == 1:
                return values[0]
    lowered = str(path).lower()
    if "test" in lowered:
        return "test"
    if "valid" in lowered or "val" in lowered:
        return "valid"
    if "train" in lowered:
        return "train"
    return "unknown"


def scan_prediction_csvs(roots: Sequence[Path], corr_target: float, mae_target: float,
                         expected_n: int, include_nontest: bool) -> List[dict]:
    rows: List[dict] = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.csv"):
            resolved = str(path.resolve())
            if resolved in seen or is_generated(path):
                continue
            seen.add(resolved)
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            label_col = find_label_column(frame)
            if label_col is None:
                continue
            split = infer_split(frame, path)
            if not include_nontest and split in {"train", "valid", "validation", "val"}:
                continue
            predictions = candidate_prediction_columns(frame, label_col)
            if not predictions:
                continue
            label = pd.to_numeric(frame[label_col], errors="coerce").to_numpy(float)
            for pred_col in predictions:
                pred = pd.to_numeric(frame[pred_col], errors="coerce").to_numpy(float)
                finite_n = int(np.sum(np.isfinite(label) & np.isfinite(pred)))
                if expected_n > 0 and finite_n != expected_n:
                    continue
                try:
                    metrics = regression_metrics(pred, label)
                except Exception:
                    continue
                dc = abs(metrics["corr"] - corr_target)
                dm = abs(metrics["mae"] - mae_target)
                rows.append({
                    "file": str(path),
                    "split": split,
                    "label_column": label_col,
                    "prediction_column": pred_col,
                    "metrics": metrics,
                    "corr_abs_diff": dc,
                    "mae_abs_diff": dm,
                    "max_pair_diff": max(dc, dm),
                    "mean_pair_diff": 0.5 * (dc + dm),
                    "display_match_both": rounded_match(metrics["corr"], corr_target)
                    and rounded_match(metrics["mae"], mae_target),
                })
    rows.sort(key=lambda x: (x["max_pair_diff"], x["mean_pair_diff"], x["file"], x["prediction_column"]))
    return rows


def numeric_pattern(value: float) -> re.Pattern:
    # Match either 0.802... or .802... for the requested three-decimal prefix.
    text = f"{value:.3f}"
    if text.startswith("0"):
        core = re.escape(text[1:])
        return re.compile(rf"(?<!\d)(?:0)?{core}\d*(?!\d)")
    return re.compile(re.escape(text))


def scan_text_contexts(roots: Sequence[Path], corr_target: float, mae_target: float,
                       context_lines: int) -> List[dict]:
    corr_re = numeric_pattern(corr_target)
    mae_re = numeric_pattern(mae_target)
    hits: List[dict] = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or is_generated(path):
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = text.splitlines()
            corr_indices = [i for i, line in enumerate(lines) if corr_re.search(line)]
            mae_indices = [i for i, line in enumerate(lines) if mae_re.search(line)]
            if not corr_indices or not mae_indices:
                continue
            pairs = []
            for i in corr_indices:
                for j in mae_indices:
                    if abs(i - j) <= context_lines:
                        pairs.append((i, j))
            if not pairs:
                continue
            # Deduplicate overlapping local windows.
            windows = set()
            for i, j in pairs:
                start = max(0, min(i, j) - context_lines)
                end = min(len(lines), max(i, j) + context_lines + 1)
                windows.add((start, end))
            for start, end in sorted(windows):
                hits.append({
                    "file": str(path),
                    "start_line": start + 1,
                    "end_line": end,
                    "context": "\n".join(f"{k+1}: {lines[k]}" for k in range(start, end)),
                })
    return hits


def show_predictions(rows: List[dict], top_k: int, corr_target: float, mae_target: float):
    exact = [x for x in rows if x["display_match_both"]]
    print(f"\n== Prediction CSVs recomputing to displayed Corr={corr_target:.3f} AND MAE={mae_target:.3f} ({len(exact)}) ==")
    if not exact:
        print("  (none found)")
    for rank, item in enumerate(exact[:top_k], 1):
        m = item["metrics"]
        print(f"{rank:2d}. {item['file']} [{item['prediction_column']}] split={item['split']} n={m['n']}")
        print(f"    Acc7={m['acc7']*100:.2f} Acc5={m['acc5']*100:.2f} Acc2={m['acc2']*100:.2f} F1={m['f1']*100:.2f} Corr={m['corr']:.6f} MAE={m['mae']:.6f}")

    print(f"\n== Closest recomputed prediction rows ({len(rows)}) ==")
    for rank, item in enumerate(rows[:top_k], 1):
        m = item["metrics"]
        print(f"{rank:2d}. dCorr={item['corr_abs_diff']:.6f} dMAE={item['mae_abs_diff']:.6f} {item['file']} [{item['prediction_column']}] split={item['split']} n={m['n']}")
        print(f"    Acc7={m['acc7']*100:.2f} Acc5={m['acc5']*100:.2f} Acc2={m['acc2']*100:.2f} F1={m['f1']*100:.2f} Corr={m['corr']:.6f} MAE={m['mae']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Trace MOSI main-table metric provenance from logs and prediction CSVs")
    parser.add_argument("--roots", nargs="+", default=["result", "log"])
    parser.add_argument("--corr", type=float, default=0.802)
    parser.add_argument("--mae", type=float, default=0.693)
    parser.add_argument("--expected-n", type=int, default=EXPECTED_MOSI_TEST_N)
    parser.add_argument("--context-lines", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--include-nontest", action="store_true")
    parser.add_argument("--write-report", default="result/mosi_main_table_provenance_trace.json")
    args = parser.parse_args()

    roots = [Path(x) for x in args.roots]
    predictions = scan_prediction_csvs(
        roots, args.corr, args.mae, args.expected_n, args.include_nontest
    )
    text_hits = scan_text_contexts(roots, args.corr, args.mae, args.context_lines)

    print(f"Scanned roots: {', '.join(map(str, roots))}")
    print(f"Expected sample count for prediction reconstruction: {args.expected_n}")
    print(f"\n== Text/log contexts containing both Corr~{args.corr:.3f} and MAE~{args.mae:.3f} ({len(text_hits)}) ==")
    if not text_hits:
        print("  (none found)")
    for rank, hit in enumerate(text_hits[:args.top_k], 1):
        print(f"\n{rank:2d}. {hit['file']} lines {hit['start_line']}-{hit['end_line']}")
        print(hit["context"])

    show_predictions(predictions, args.top_k, args.corr, args.mae)

    report = {
        "target": {"Corr": args.corr, "MAE": args.mae},
        "roots": [str(x) for x in roots],
        "expected_n": args.expected_n,
        "text_hits": text_hits,
        "prediction_exact_display_matches": [x for x in predictions if x["display_match_both"]],
        "prediction_closest": predictions[: max(1, args.top_k)],
    }
    output = Path(args.write_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote report: {output}")


if __name__ == "__main__":
    main()
