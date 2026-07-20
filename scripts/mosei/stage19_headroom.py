"""Label-free Stage 19 recoverability headroom audit from frozen caches."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.mgrd_utils import (
    BOUNDARIES,
    GRANULARITIES,
    MISSING_MODES,
    TAU,
    ordinal_probability,
    recover_evaluator_decisions,
    recoverability,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(1111, 1114))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def main():
    cli = parse_args()
    root = Path(cli.cache_root) / "seed{}".format(cli.seed)
    manifest_path = root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["entries"]["train"]
    cache_path = Path(entry["path"])
    if sha256_file(cache_path) != entry["sha256"]:
        raise RuntimeError("Headroom cache SHA mismatch.")
    with np.load(cache_path, allow_pickle=False) as archive:
        teacher = archive["teacher_lav"].astype(np.float32)
        references = {mode: archive["moddrop_{}".format(mode)].astype(np.float32) for mode in MISSING_MODES}

    rows = []
    mode_summary = {}
    all_c = []
    all_coarse = []
    granularity_coverages = {granularity: [] for granularity in GRANULARITIES}
    for mode in MISSING_MODES:
        reference = references[mode]
        gap = np.abs(teacher - reference)
        stable = {
            granularity: recover_evaluator_decisions(teacher, granularity)
            == recover_evaluator_decisions(reference, granularity)
            for granularity in GRANULARITIES
        }
        coarse_only = stable["Acc2"] & (~stable["Acc5"] | ~stable["Acc7"])
        polarity_stable_acc5_unstable = stable["Acc2"] & ~stable["Acc5"]
        acc5_stable_acc7_unstable = stable["Acc5"] & ~stable["Acc7"]
        gap_threshold = float(np.median(gap))
        acc7_stable_large_gap = stable["Acc7"] & (gap > gap_threshold)
        all_stable = stable["Acc2"] & stable["Acc5"] & stable["Acc7"]
        all_unstable = ~stable["Acc2"] & ~stable["Acc5"] & ~stable["Acc7"]
        all_coarse.append(coarse_only.astype(np.float32))

        summary = {
            "sample_count": int(teacher.size),
            "teacher_reference_abs_gap_mean": float(gap.mean()),
            "teacher_reference_abs_gap_median": float(np.median(gap)),
            "coarse_only_headroom": float(coarse_only.mean()),
            "polarity_stable_acc5_unstable": float(polarity_stable_acc5_unstable.mean()),
            "acc5_stable_acc7_unstable": float(acc5_stable_acc7_unstable.mean()),
            "acc7_stable_scalar_gap_above_mode_median": float(acc7_stable_large_gap.mean()),
            "all_granularities_stable": float(all_stable.mean()),
            "all_granularities_unstable": float(all_unstable.mean()),
        }
        for granularity in GRANULARITIES:
            boundaries = BOUNDARIES[granularity]
            q_teacher = ordinal_probability(torch.from_numpy(teacher), boundaries, TAU)
            q_reference = ordinal_probability(torch.from_numpy(reference), boundaries, TAU)
            weights = recoverability(q_teacher, q_reference).numpy()
            flat = weights.reshape(-1)
            all_c.append(flat)
            coverage = float((flat > 0).mean())
            granularity_coverages[granularity].append(coverage)
            summary["{}_boundary_agreement".format(granularity)] = float(stable[granularity].mean())
            summary["{}_c_mean".format(granularity)] = float(flat.mean())
            summary["{}_c_std".format(granularity)] = float(flat.std())
            summary["{}_c_nonzero_rate".format(granularity)] = coverage
            summary["{}_c_cv".format(granularity)] = float(flat.std() / max(flat.mean(), 1e-12))
            summary["{}_c_ess".format(granularity)] = float(
                flat.sum() ** 2 / max(np.square(flat).sum(), 1e-12)
            )
            for quantile in (0, 0.25, 0.5, 0.75, 0.9, 1):
                summary["{}_c_q{:02d}".format(granularity, int(quantile * 100))] = float(
                    np.quantile(flat, quantile)
                )
            rows.append(
                {
                    "Mode": mode,
                    "Granularity": granularity,
                    "BoundaryCount": len(boundaries),
                    "Agreement": summary["{}_boundary_agreement".format(granularity)],
                    "CMean": summary["{}_c_mean".format(granularity)],
                    "CStd": summary["{}_c_std".format(granularity)],
                    "CNonzeroRate": coverage,
                    "CCV": summary["{}_c_cv".format(granularity)],
                    "EffectiveSampleSize": summary["{}_c_ess".format(granularity)],
                    "CoarseOnlyHeadroom": summary["coarse_only_headroom"],
                    "TeacherReferenceGapMean": summary["teacher_reference_abs_gap_mean"],
                }
            )
        mode_summary[mode] = summary

    combined_c = np.concatenate(all_c)
    aggregate_coarse = float(np.concatenate(all_coarse).mean())
    aggregate_cv = float(combined_c.std() / max(combined_c.mean(), 1e-12))
    modes_above_five = sum(
        value["coarse_only_headroom"] >= 0.05 for value in mode_summary.values()
    )
    coverage_pass = all(
        min(values) >= 0.10 for values in granularity_coverages.values()
    )
    passed = (
        aggregate_coarse >= 0.10
        and modes_above_five >= 2
        and aggregate_cv >= 0.10
        and coverage_pass
    )
    result = {
        "status": "STAGE19C_RECOVERABILITY_HEADROOM_PRESENT"
        if passed
        else "STAGE19C_RECOVERABILITY_HEADROOM_ABSENT",
        "seed": cli.seed,
        "tau": TAU,
        "boundaries": {key: list(value) for key, value in BOUNDARIES.items()},
        "aggregate_coarse_only_headroom": aggregate_coarse,
        "modes_with_coarse_only_headroom_at_least_0_05": modes_above_five,
        "aggregate_c_mean": float(combined_c.mean()),
        "aggregate_c_std": float(combined_c.std()),
        "aggregate_c_cv": aggregate_cv,
        "minimum_nonzero_coverage_by_granularity": {
            key: min(values) for key, values in granularity_coverages.items()
        },
        "gate": {
            "aggregate_coarse_only_min": 0.10,
            "mode_count_min": 2,
            "per_mode_coarse_only_min": 0.05,
            "c_cv_min": 0.10,
            "per_mode_granularity_nonzero_coverage_min": 0.10,
            "passed": passed,
        },
        "per_mode": mode_summary,
        "uses_ground_truth": False,
        "locked_test_access_count": 0,
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "granularity_headroom.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    frame = pd.DataFrame(rows)
    atomic_text(output / "per_mode_granularity.tsv", frame.to_csv(sep="\t", index=False))
    lines = [
        "# Stage 19C Recoverability Headroom",
        "",
        "- Status: `{}`".format(result["status"]),
        "- Aggregate coarse-only headroom: {:.4%}".format(aggregate_coarse),
        "- Aggregate c coefficient of variation: {:.4f}".format(aggregate_cv),
        "- Modes above 5% coarse-only headroom: {}/3".format(modes_above_five),
        "- Ground-truth used: no",
        "- Locked Test access count: 0",
        "",
        "| Mode | Coarse-only | Acc2 agree | Acc5 agree | Acc7 agree | Gap mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode, values in mode_summary.items():
        lines.append(
            "| {} | {:.4%} | {:.4%} | {:.4%} | {:.4%} | {:.6f} |".format(
                mode,
                values["coarse_only_headroom"],
                values["Acc2_boundary_agreement"],
                values["Acc5_boundary_agreement"],
                values["Acc7_boundary_agreement"],
                values["teacher_reference_abs_gap_mean"],
            )
        )
    atomic_text(output / "granularity_headroom.md", "\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
