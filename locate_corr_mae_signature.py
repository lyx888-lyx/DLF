"""Locate historical result records near the MOSI Ours Corr/MAE signature.

Read-only diagnostic. Searches CSV/JSON artifacts under result/ for records whose
Corr and MAE are closest to target values (default Corr=.802, MAE=.693).
Post-hoc analysis artifacts and obvious target/expected audit objects are excluded
by default so the script reports historical experiment evidence rather than the
query itself.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def norm_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def to_float(value: Any):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def extract_pair(mapping: Dict[str, Any]):
    normalized = {norm_key(k): v for k, v in mapping.items()}
    corr = None
    mae = None
    for key in ("lavcorr", "corr", "lavcorrelation", "correlation"):
        if key in normalized:
            corr = to_float(normalized[key])
            if corr is not None:
                break
    for key in ("lavmae", "mae"):
        if key in normalized:
            mae = to_float(normalized[key])
            if mae is not None:
                break
    if corr is None or mae is None:
        return None
    return float(corr), float(mae)


def json_objects(value: Any, path: str = "$") -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from json_objects(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from json_objects(child, f"{path}[{i}]")


def is_excluded_location(path: Path, location: str, include_posthoc: bool) -> bool:
    p = str(path).replace("\\", "/").lower()
    loc = str(location).lower()
    if not include_posthoc and "/posthoc_analysis/" in f"/{p}":
        return True
    forbidden = ("expected", "target", "signature", "paper_table_audit")
    return any(token in loc for token in forbidden)


def context_from_mapping(mapping: Dict[str, Any]) -> str:
    labels = []
    for key in ("Method", "method", "Name", "name", "Mode", "mode", "Split", "split", "Seed", "seed"):
        if key in mapping:
            value = mapping[key]
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            labels.append(f"{key}={value}")
    return "; ".join(labels)


def scan_csv(path: Path):
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    out = []
    for idx, row in frame.iterrows():
        mapping = row.to_dict()
        pair = extract_pair(mapping)
        if pair is not None:
            out.append((f"row={idx}", context_from_mapping(mapping), pair))
    return out


def scan_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for location, mapping in json_objects(value):
        pair = extract_pair(mapping)
        if pair is not None:
            out.append((location, context_from_mapping(mapping), pair))
    return out


def main():
    parser = argparse.ArgumentParser(description="Find historical records near a Corr/MAE signature")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--corr", type=float, default=0.802)
    parser.add_argument("--mae", type=float, default=0.693)
    parser.add_argument("--display-tol", type=float, default=0.0005,
                        help="Tolerance for a value to display as the requested 3-decimal target")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--include-posthoc", action="store_true")
    parser.add_argument("--write-report", default="result/corr_mae_signature_matches.json")
    args = parser.parse_args()

    root = Path(args.result_root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    candidates: List[dict] = []
    files = sorted(list(root.rglob("*.csv")) + list(root.rglob("*.json")))
    for path in files:
        rows = scan_csv(path) if path.suffix.lower() == ".csv" else scan_json(path)
        for location, context, (corr, mae) in rows:
            if is_excluded_location(path, location, args.include_posthoc):
                continue
            dc = abs(corr - args.corr)
            dm = abs(mae - args.mae)
            candidates.append({
                "file": str(path),
                "location": location,
                "context": context,
                "corr": corr,
                "mae": mae,
                "corr_abs_diff": dc,
                "mae_abs_diff": dm,
                "max_pair_diff": max(dc, dm),
                "mean_pair_diff": 0.5 * (dc + dm),
                "display_match_corr": dc < args.display_tol,
                "display_match_mae": dm < args.display_tol,
                "display_match_both": dc < args.display_tol and dm < args.display_tol,
            })

    candidates.sort(key=lambda x: (x["max_pair_diff"], x["mean_pair_diff"], x["file"], x["location"]))
    both = [x for x in candidates if x["display_match_both"]]
    corr_only = [x for x in candidates if x["display_match_corr"] and not x["display_match_mae"]]
    mae_only = [x for x in candidates if x["display_match_mae"] and not x["display_match_corr"]]

    def show(title: str, rows: List[dict], limit: int):
        print(f"\n== {title} ({len(rows)}) ==")
        if not rows:
            print("  (none found)")
            return
        for rank, item in enumerate(rows[:limit], 1):
            print(f"{rank:2d}. Corr={item['corr']:.9f} MAE={item['mae']:.9f} "
                  f"dCorr={item['corr_abs_diff']:.9f} dMAE={item['mae_abs_diff']:.9f}")
            print(f"    {item['file']}  [{item['location']}] {item['context']}")

    show(f"Historical records matching displayed Corr={args.corr:.3f} AND MAE={args.mae:.3f}", both, args.top_k)
    show(f"Historical records matching displayed Corr={args.corr:.3f} only", corr_only, min(args.top_k, 20))
    show(f"Historical records matching displayed MAE={args.mae:.3f} only", mae_only, min(args.top_k, 20))
    show("Closest historical Corr/MAE pairs overall", candidates, args.top_k)

    report = {
        "target": {"Corr": args.corr, "MAE": args.mae, "display_tolerance": args.display_tol},
        "scanned_files": len(files),
        "candidate_pairs": len(candidates),
        "display_match_both": both,
        "display_match_corr_only": corr_only,
        "display_match_mae_only": mae_only,
        "closest": candidates[: max(1, args.top_k)],
        "posthoc_excluded": not args.include_posthoc,
    }
    output = Path(args.write_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote report: {output}")


if __name__ == "__main__":
    main()
