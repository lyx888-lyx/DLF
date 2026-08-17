"""Discover reusable prediction artifacts for sentiment-region robustness analysis.

This script is intentionally read-only and offline.  It recursively scans a local
result directory for CSV/JSON files, inspects only lightweight metadata, and
reports which artifacts can be used directly by
``analyze_sentiment_region_missing_robustness.py``.

It never constructs a dataset loader, never runs a model forward pass, and never
changes any result file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


MODES: Tuple[str, ...] = ("LAV", "LA", "LV", "L")
PRED_COLUMNS: Tuple[str, ...] = tuple(f"{m}_pred" for m in MODES)
IDENTITY_COLUMNS: Tuple[str, ...] = ("sample_index", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only discovery of prediction artifacts for post-hoc analysis"
    )
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--dataset", choices=("mosi", "mosei", "all"), default="all")
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional JSON report path; does not modify model/result artifacts",
    )
    return parser.parse_args()


def infer_dataset(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    if "mosei" in text:
        return "mosei"
    if "mosi" in text:
        return "mosi"
    return "unknown"


def infer_split(path: Path, columns: Sequence[str], frame: Optional[pd.DataFrame]) -> str:
    text = path.name.lower()
    if "test" in text:
        return "test"
    if "valid" in text or "val" in text:
        return "valid"
    if frame is not None and "Split" in columns:
        values = frame["Split"].dropna().astype(str).str.lower().unique().tolist()
        if len(values) == 1 and values[0] in {"valid", "test", "train"}:
            return values[0]
    return "unknown"


def classify(path: Path, columns: Sequence[str]) -> List[str]:
    text = str(path).lower().replace("\\", "/")
    tags: List[str] = []
    column_set = set(columns)
    if set(IDENTITY_COLUMNS).issubset(column_set) and "LAV_pred" in column_set:
        tags.append("sample_prediction_csv")
    if all(col in column_set for col in PRED_COLUMNS):
        tags.append("all_mode_predictions")
    elif "LAV_pred" in column_set:
        tags.append("lav_only_predictions")

    if "cfcompat_prediction_ensemble_v1" in text:
        if "ensemble_predictions" in path.name.lower():
            tags.append("raw5_ensemble")
        elif "online_seed" in path.name.lower():
            tags.append("raw5_member")
        else:
            tags.append("cfcompat_prediction_ensemble")
    if "cf_compat_kd_v1" in text or "mosei_cfcompat_v1" in text:
        tags.append("cfcompat")
    if "normal" in text or "clean_stage0" in text:
        tags.append("clean_or_normal_candidate")
    if "moddrop" in text:
        tags.append("moddrop_candidate")
    if "fixedblend" in text:
        tags.append("fixedblend")
    if "v13" in text:
        tags.append("v13")
    if "complementarity" in text or "hybrid" in text:
        tags.append("complementary_or_hybrid")
    return tags


def inspect_csv(path: Path) -> Dict[str, object]:
    record: Dict[str, object] = {
        "path": str(path),
        "suffix": ".csv",
        "dataset": infer_dataset(path),
    }
    try:
        frame = pd.read_csv(path, nrows=8)
    except Exception as exc:  # discovery should continue after one bad artifact
        record.update({"readable": False, "error": repr(exc)})
        return record
    columns = [str(c) for c in frame.columns]
    record.update(
        {
            "readable": True,
            "columns": columns,
            "split": infer_split(path, columns, frame),
            "tags": classify(path, columns),
            "has_sample_predictions": bool(
                set(IDENTITY_COLUMNS).issubset(set(columns)) and "LAV_pred" in columns
            ),
            "available_modes": [m for m in MODES if f"{m}_pred" in columns],
        }
    )
    for field in ("Method", "Seed", "Split", "SelectedBy"):
        if field in frame.columns:
            values = frame[field].dropna().astype(str).unique().tolist()[:8]
            record[f"{field}_values"] = values
    return record


def inspect_json(path: Path) -> Dict[str, object]:
    record: Dict[str, object] = {
        "path": str(path),
        "suffix": ".json",
        "dataset": infer_dataset(path),
        "readable": True,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        record.update({"readable": False, "error": repr(exc)})
        return record
    text = str(path).lower().replace("\\", "/")
    tags: List[str] = []
    if "fixedblend" in text:
        tags.append("fixedblend_metadata")
    if "v13" in text:
        tags.append("v13_metadata")
    if "summary" in path.name.lower():
        tags.append("summary")
    record["tags"] = tags
    if isinstance(payload, dict):
        record["top_level_keys"] = sorted(map(str, payload.keys()))[:50]
    return record


def priority(record: Dict[str, object]) -> Tuple[int, str]:
    tags = set(record.get("tags", []))
    score = 0
    if "sample_prediction_csv" in tags:
        score += 100
    if "all_mode_predictions" in tags:
        score += 40
    if "raw5_ensemble" in tags:
        score += 35
    if "clean_or_normal_candidate" in tags:
        score += 30
    if "cfcompat" in tags:
        score += 20
    if "v13" in tags:
        score += 15
    if "fixedblend" in tags:
        score += 10
    return (-score, str(record.get("path", "")))


def candidate_groups(records: Iterable[Dict[str, object]], dataset: str) -> Dict[str, List[Dict[str, object]]]:
    filtered = [
        r
        for r in records
        if r.get("readable")
        and r.get("suffix") == ".csv"
        and r.get("has_sample_predictions")
        and (dataset == "all" or r.get("dataset") in {dataset, "unknown"})
    ]
    groups = {
        "baseline_candidates": [],
        "raw5_candidates": [],
        "cfcompat_candidates": [],
        "v13_candidates": [],
        "other_prediction_candidates": [],
    }
    for record in sorted(filtered, key=priority):
        tags = set(record.get("tags", []))
        if "clean_or_normal_candidate" in tags:
            groups["baseline_candidates"].append(record)
        if "raw5_ensemble" in tags:
            groups["raw5_candidates"].append(record)
        if "cfcompat" in tags or "raw5_member" in tags:
            groups["cfcompat_candidates"].append(record)
        if "v13" in tags:
            groups["v13_candidates"].append(record)
        if not tags.intersection(
            {"clean_or_normal_candidate", "raw5_ensemble", "cfcompat", "raw5_member", "v13"}
        ):
            groups["other_prediction_candidates"].append(record)
    return groups


def short(record: Dict[str, object]) -> str:
    modes = ",".join(record.get("available_modes", [])) or "none"
    tags = ",".join(record.get("tags", [])) or "-"
    split = record.get("split", "unknown")
    return f"{record['path']}  [split={split}; modes={modes}; tags={tags}]"


def print_group(title: str, records: Sequence[Dict[str, object]], limit: int = 20) -> None:
    print(f"\n== {title} ({len(records)}) ==")
    if not records:
        print("  (none found)")
        return
    for record in records[:limit]:
        print("  " + short(record))
    if len(records) > limit:
        print(f"  ... {len(records) - limit} more")


def main() -> None:
    args = parse_args()
    root = Path(args.result_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Result root does not exist: {root}")

    paths = sorted(root.rglob("*"))
    files = [p for p in paths if p.is_file() and p.suffix.lower() in {".csv", ".json"}]
    if len(files) > int(args.max_files):
        raise RuntimeError(
            f"Refusing to scan {len(files)} files (> --max-files={args.max_files}); "
            "increase the limit explicitly if intended."
        )

    records: List[Dict[str, object]] = []
    for path in files:
        if path.suffix.lower() == ".csv":
            records.append(inspect_csv(path))
        else:
            records.append(inspect_json(path))

    groups = candidate_groups(records, args.dataset)
    print(f"Scanned {len(files)} CSV/JSON files under {root}")
    print_group("Likely clean/normal DLF baseline predictions", groups["baseline_candidates"])
    print_group("Raw5 ensemble predictions", groups["raw5_candidates"])
    print_group("CFCompat member/other CFCompat predictions", groups["cfcompat_candidates"])
    print_group("v13 sample-level prediction candidates", groups["v13_candidates"])
    print_group("Other sample-level prediction candidates", groups["other_prediction_candidates"])

    all_sample = [r for r in records if r.get("has_sample_predictions")]
    fixedblend_meta = [
        r
        for r in records
        if "fixedblend_metadata" in set(r.get("tags", []))
        or "fixedblend" in set(r.get("tags", []))
    ]
    print(f"\nSample-level prediction CSVs found: {len(all_sample)}")
    print(f"FixedBlend-related CSV/JSON artifacts found: {len(fixedblend_meta)}")

    if groups["baseline_candidates"] and groups["raw5_candidates"]:
        print("\nA baseline + Raw5 analysis appears possible immediately.")
        print("Use the paths above with analyze_sentiment_region_missing_robustness.py.")
    else:
        print("\nNo automatic baseline + Raw5 pair was confidently identified yet.")

    if groups["v13_candidates"]:
        print("A sample-level v13 artifact exists, so an offline final blend may be reconstructable.")
    else:
        print(
            "No sample-level v13 prediction CSV was identified. If your final Ours uses Raw5+v13, "
            "the final sample-level Ours prediction cannot be reconstructed offline from CSVs alone; "
            "we should first inspect/replay the frozen v13 source explicitly."
        )

    report = {
        "result_root": str(root),
        "dataset_filter": args.dataset,
        "records": records,
        "groups": groups,
    }
    if args.write_report:
        output = Path(args.write_report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote discovery report: {output}")


if __name__ == "__main__":
    main()
