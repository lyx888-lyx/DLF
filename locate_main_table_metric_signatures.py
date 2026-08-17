"""Locate aggregate result artifacts that best match the paper-table MOSI rows.

This is a read-only diagnostic. It recursively scans CSV/JSON files under a result
root and ranks rows/objects by distance to two frozen target signatures:

DLF  : Acc7 47.08, Acc5 52.33, Acc2 85.06, F1 85.04, Corr .781, MAE .731
Ours : Acc7 49.47, Acc5 55.10, Acc2 85.06, F1 85.01, Corr .802, MAE .693

The purpose is provenance only: identify which saved aggregate artifact/method
most likely produced the numbers already placed in the table. It never selects,
trains, tunes, or changes a model.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

TARGETS = {
    "DLF": {"acc7": .4708, "acc5": .5233, "acc2": .8506, "f1": .8504, "corr": .781, "mae": .731},
    "Ours": {"acc7": .4947, "acc5": .5510, "acc2": .8506, "f1": .8501, "corr": .802, "mae": .693},
}

ALIASES = {
    "acc7": ("lavacc7", "acc7", "lavaccuracy7", "accuracy7"),
    "acc5": ("lavacc5", "acc5", "lavaccuracy5", "accuracy5"),
    "acc2": ("lavacc2", "acc2", "lavaccuracy2", "accuracy2"),
    "f1": ("lavf1score", "lavf1", "f1score", "f1"),
    "corr": ("lavcorr", "corr", "lavcorrelation", "correlation"),
    "mae": ("lavmae", "mae"),
}


def norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def to_float(value: Any):
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def normalize_metric(metric: str, value: float) -> float:
    # Accuracy/F1 values are encountered both as 0..1 and 0..100.
    if metric in {"acc7", "acc5", "acc2", "f1"} and abs(value) > 1.5:
        return value / 100.0
    return value


def extract_metrics(mapping: Dict[str, Any]) -> Dict[str, float]:
    normalized = {norm_key(k): v for k, v in mapping.items()}
    out: Dict[str, float] = {}
    for metric, aliases in ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                value = to_float(normalized[alias])
                if value is not None:
                    out[metric] = normalize_metric(metric, value)
                    break
    return out


def score(metrics: Dict[str, float], target: Dict[str, float]):
    common = [key for key in target if key in metrics]
    if len(common) < 4:
        return None
    diffs = {key: abs(metrics[key] - target[key]) for key in common}
    # Primary rank: maximum discrepancy, then mean discrepancy, then prefer more metrics.
    return max(diffs.values()), sum(diffs.values()) / len(diffs), -len(common), diffs


def json_objects(value: Any, path: str = "$") -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from json_objects(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from json_objects(child, f"{path}[{i}]")


def scan_csv(path: Path) -> List[Dict[str, Any]]:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    results = []
    for idx, row in frame.iterrows():
        mapping = row.to_dict()
        metrics = extract_metrics(mapping)
        if len(metrics) >= 4:
            labels = []
            for key in ("Method", "method", "Name", "name", "Mode", "mode", "Split", "split", "Seed", "seed"):
                if key in mapping and pd.notna(mapping[key]):
                    labels.append(f"{key}={mapping[key]}")
            results.append({"location": f"row={idx}", "context": "; ".join(labels), "metrics": metrics})
    return results


def scan_json(path: Path) -> List[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    results = []
    for obj_path, mapping in json_objects(value):
        metrics = extract_metrics(mapping)
        if len(metrics) >= 4:
            labels = []
            for key in ("Method", "method", "Name", "name", "Mode", "mode", "Split", "split", "Seed", "seed"):
                if key in mapping:
                    labels.append(f"{key}={mapping[key]}")
            results.append({"location": obj_path, "context": "; ".join(labels), "metrics": metrics})
    return results


def main():
    parser = argparse.ArgumentParser(description="Locate result artifacts matching main-table MOSI metric signatures")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--write-report", default="result/main_table_metric_signature_matches.json")
    args = parser.parse_args()

    root = Path(args.result_root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    candidates = []
    files = sorted(list(root.rglob("*.csv")) + list(root.rglob("*.json")))
    for path in files:
        rows = scan_csv(path) if path.suffix.lower() == ".csv" else scan_json(path)
        for row in rows:
            row["file"] = str(path)
            candidates.append(row)

    report = {"scanned_files": len(files), "candidate_objects": len(candidates), "targets": TARGETS, "matches": {}}
    for target_name, target in TARGETS.items():
        ranked = []
        for candidate in candidates:
            scored = score(candidate["metrics"], target)
            if scored is None:
                continue
            max_diff, mean_diff, neg_count, diffs = scored
            ranked.append({
                **candidate,
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "metric_count": -neg_count,
                "per_metric_abs_diff": diffs,
            })
        ranked.sort(key=lambda x: (x["max_abs_diff"], x["mean_abs_diff"], -x["metric_count"], x["file"], x["location"]))
        top = ranked[: max(1, int(args.top_k))]
        report["matches"][target_name] = top
        print(f"\n== Closest matches to {target_name} main-table row ==")
        for rank, item in enumerate(top, 1):
            m = item["metrics"]
            pretty = ", ".join(f"{k}={m[k]:.6f}" for k in sorted(m))
            print(f"{rank:2d}. maxDiff={item['max_abs_diff']:.6f} meanDiff={item['mean_abs_diff']:.6f} n={item['metric_count']}")
            print(f"    {item['file']}  [{item['location']}] {item['context']}")
            print(f"    {pretty}")

    out = Path(args.write_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote report: {out}")


if __name__ == "__main__":
    main()
