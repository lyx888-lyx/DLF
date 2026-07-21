"""Stage 21A phases 0--5: binding, pairs, frozen caches, ICC, and splits."""

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.missing_utils import MissingModalityWrapper, mode_to_mask
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed

from scripts.mosei.stage21a_common import (
    ALLOWED_SPLITS,
    MODES,
    atomic_frame,
    atomic_json,
    atomic_npz,
    atomic_text,
    assert_unique_complete_ids,
    balanced_within_pairs,
    canonical_ids,
    group_indices,
    icc_permutation_and_bootstrap,
    label_histogram,
    make_source_split,
    one_way_icc_from_groups,
    ordered_id_sha,
    parse_sample_id,
    require_allowed_split,
    residual_structure,
    select_delta,
    sha256_file,
    sha256_json,
    size_summary,
    unordered_id_sha,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--uniform-checkpoint", required=True)
    parser.add_argument("--uniform-pointer", required=True)
    parser.add_argument("--uniform-run-manifest", required=True)
    parser.add_argument("--uniform-epoch-metrics", required=True)
    parser.add_argument("--teacher-cache-manifest", required=True)
    parser.add_argument("--compatibility-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-id", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=2000)
    return parser.parse_args()


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_args(cli):
    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = 1111
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def locked_dataset(args, split):
    require_allowed_split(split)
    return MMDataset(args, mode=split)


def dataset_table(dataset):
    ids = canonical_ids(dataset.ids)
    labels = np.asarray(dataset.labels["M"], dtype=np.float32).reshape(-1)
    parsed = assert_unique_complete_ids(ids)
    video_ids = np.asarray([value[0] for value in parsed], dtype=object)
    clip_ids = np.asarray([value[1] for value in parsed], dtype=object)
    return {
        "ids": np.asarray(ids, dtype=object),
        "video_ids": video_ids,
        "clip_ids": clip_ids,
        "labels": labels,
    }


def split_binding_summary(name, table):
    ids = table["ids"]
    video_ids = table["video_ids"]
    clip_ids = table["clip_ids"]
    labels = table["labels"]
    groups = group_indices(video_ids)
    pairs = list(zip(video_ids.tolist(), clip_ids.tolist()))
    sizes = [len(value) for value in groups.values()]
    return {
        "split": name,
        "sample_count": int(len(ids)),
        "unique_video_id_count": int(len(groups)),
        "clips_per_video": size_summary(sizes),
        "duplicate_video_clip_count": int(len(pairs) - len(set(pairs))),
        "duplicate_sample_id_count": int(len(ids) - len(set(ids.tolist()))),
        "missing_video_id_count": int(sum(not str(value) for value in video_ids)),
        "missing_clip_id_count": int(sum(not str(value) for value in clip_ids)),
        "label_min": float(labels.min()),
        "label_max": float(labels.max()),
        "label_missing_count": int(np.isnan(labels).sum()),
        "label_nonfinite_count": int((~np.isfinite(labels)).sum()),
        "label_out_of_range_count": int(np.sum((labels < -3) | (labels > 3))),
        "ordered_sample_id_sha256": ordered_id_sha(ids),
        "unordered_sample_id_sha256": unordered_id_sha(ids),
    }


def data_binding(cli, datasets, tables, output):
    cache = json.loads(Path(cli.teacher_cache_manifest).read_text())
    summaries = {split: split_binding_summary(split, tables[split]) for split in ALLOWED_SPLITS}
    train_ids = set(tables["train"]["ids"].tolist())
    valid_ids = set(tables["valid"]["ids"].tolist())
    train_videos = set(tables["train"]["video_ids"].tolist())
    valid_videos = set(tables["valid"]["video_ids"].tolist())
    report = {
        "created_at": utc_now(),
        "feature_path": str(datasets["train"].args.featurePath),
        "sample_id_format": "video_id$_$clip_id",
        "video_id_source": "prefix of the official pkl id field before $_$",
        "clip_id_source": "suffix of the official pkl id field after $_$",
        "video_id_semantics": "original source video ID",
        "speaker_id_available": False,
        "speaker_id_claimed": False,
        "splits": summaries,
        "train_valid_video_overlap_count": int(len(train_videos & valid_videos)),
        "train_valid_sample_overlap_count": int(len(train_ids & valid_ids)),
        "teacher_cache_manifest": str(cli.teacher_cache_manifest),
        "teacher_cache_manifest_sha256": sha256_file(cli.teacher_cache_manifest),
        "cache_binding": {},
        "locked_test_access_count": 0,
    }
    failures = []
    for split in ALLOWED_SPLITS:
        expected = cache["entries"][split]
        actual = summaries[split]
        match = (
            actual["ordered_sample_id_sha256"] == expected["ordered_sample_id_sha256"]
            and actual["unordered_sample_id_sha256"] == expected["unordered_sample_id_sha256"]
            and actual["sample_count"] == expected["sample_count"]
        )
        report["cache_binding"][split] = {
            "matched": bool(match),
            "expected_order_sha256": expected["ordered_sample_id_sha256"],
            "actual_order_sha256": actual["ordered_sample_id_sha256"],
        }
        if not match:
            failures.append("{} sample/cache binding".format(split))
        for key in (
            "duplicate_video_clip_count",
            "duplicate_sample_id_count",
            "missing_video_id_count",
            "missing_clip_id_count",
            "label_missing_count",
            "label_nonfinite_count",
            "label_out_of_range_count",
        ):
            if actual[key]:
                failures.append("{} {}".format(split, key))
    if report["train_valid_video_overlap_count"] or report["train_valid_sample_overlap_count"]:
        failures.append("Train/Valid split overlap")
    report["hard_failures"] = failures
    report["status"] = "PASS" if not failures else "STAGE21A_BLOCKED_ID_BINDING_ERROR"
    atomic_json(output / "data/data_binding_audit.json", report)
    lines = [
        "# Stage 21A data binding audit",
        "",
        "- Status: `{}`".format(report["status"]),
        "- Sample ID format: `video_id$_$clip_id`",
        "- `video_id` denotes the original source video; no speaker metadata is present.",
        "- Train/Valid source overlap: **{}**".format(report["train_valid_video_overlap_count"]),
        "- Train/Valid sample overlap: **{}**".format(report["train_valid_sample_overlap_count"]),
        "- Locked Test access count: **0**",
        "",
        "| Split | Samples | Sources | Clips/source mean | median | p90 | max | Order/cache bound |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in ALLOWED_SPLITS:
        value = summaries[split]
        size = value["clips_per_video"]
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.1f} | {:.1f} | {} | {} |".format(
                split,
                value["sample_count"],
                value["unique_video_id_count"],
                size["mean"],
                size["median"],
                size["p90"],
                size["max"],
                report["cache_binding"][split]["matched"],
            )
        )
    atomic_text(output / "data/data_binding_audit.md", "\n".join(lines) + "\n")
    summary_rows = []
    for split in ALLOWED_SPLITS:
        groups = group_indices(tables[split]["video_ids"])
        labels = tables[split]["labels"]
        for video_id in sorted(groups):
            index = groups[video_id]
            summary_rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "clip_count": len(index),
                    "label_mean": float(labels[index].mean()),
                    "label_std": float(labels[index].std()),
                    "label_min": float(labels[index].min()),
                    "label_max": float(labels[index].max()),
                }
            )
    atomic_frame(output / "data/video_group_summary.tsv", summary_rows)
    sample_manifest = {
        "created_at": utc_now(),
        "locked_test_access_count": 0,
        "splits": {
            split: {
                "ordered_sample_id_sha256": summaries[split]["ordered_sample_id_sha256"],
                "unordered_sample_id_sha256": summaries[split]["unordered_sample_id_sha256"],
                "samples": [
                    {
                        "sample_id": str(sample_id),
                        "video_id": str(video_id),
                        "clip_id": str(clip_id),
                        "label": float(label),
                    }
                    for sample_id, video_id, clip_id, label in zip(
                        tables[split]["ids"],
                        tables[split]["video_ids"],
                        tables[split]["clip_ids"],
                        tables[split]["labels"],
                    )
                ],
            }
            for split in ALLOWED_SPLITS
        },
    }
    atomic_json(output / "data/sample_manifest.json", sample_manifest)
    if failures:
        raise RuntimeError("ID binding hard failure: {}".format(failures))
    return report


def pair_audit(table, output):
    selected, coverage = select_delta(table["video_ids"], table["labels"])
    report = {
        "created_at": utc_now(),
        "delta_candidates": [1.0, 0.75, 0.5],
        "selection_uses_official_train_only": True,
        "coverage": coverage,
        "selected_delta": selected,
        "locked_test_access_count": 0,
    }
    if selected is None:
        report["status"] = "STAGE21A_FAILED_INSUFFICIENT_PAIR_COVERAGE"
        atomic_json(output / "pairs/pair_coverage.json", report)
        atomic_text(output / "pairs/pair_coverage.md", "# Pair coverage\n\nNo frozen delta passed.\n")
        return report, []
    pairs = balanced_within_pairs(table["video_ids"], table["labels"], selected)
    for row in pairs:
        left, right = row["left_index"], row["right_index"]
        row.update(
            {
                "left_sample_id": str(table["ids"][left]),
                "right_sample_id": str(table["ids"][right]),
                "left_label": float(table["labels"][left]),
                "right_label": float(table["labels"][right]),
            }
        )
    report.update(
        {
            "status": "PASS",
            "selected_pair_count": len(pairs),
            "selected_pair_manifest_sha256": sha256_json(pairs),
            "max_pairs_per_video": 64,
            "per_video_total_weight_equal": True,
        }
    )
    atomic_json(output / "pairs/pair_coverage.json", report)
    atomic_frame(output / "pairs/within_video_pair_manifest.tsv", pairs)
    statistics = []
    by_video = {}
    for row in pairs:
        by_video.setdefault(row["video_id"], []).append(row)
    groups = group_indices(table["video_ids"])
    for video_id in sorted(groups):
        index = groups[video_id]
        labels = table["labels"][index]
        selected_rows = by_video.get(video_id, [])
        median = float(np.median(labels))
        statistics.append(
            {
                "video_id": video_id,
                "clip_count": len(index),
                "label_mean": float(labels.mean()),
                "label_median": median,
                "label_std": float(labels.std()),
                "label_mad": float(np.median(np.abs(labels - median))),
                "label_min": float(labels.min()),
                "label_max": float(labels.max()),
                "label_range": float(labels.max() - labels.min()),
                "positive_count": int(np.sum(labels > 0)),
                "negative_count": int(np.sum(labels < 0)),
                "neutral_count": int(np.sum(labels == 0)),
                "all_pair_count": int(len(index) * (len(index) - 1) // 2),
                "selected_pair_count": len(selected_rows),
                "selected_pair_weight_sum": float(sum(row["raw_video_balanced_weight"] for row in selected_rows)),
                "valid_pair_count_delta_1_0": len(
                    [1 for i in range(len(labels)) for j in range(i + 1, len(labels)) if abs(labels[i] - labels[j]) >= 1.0]
                ),
                "valid_pair_count_delta_0_75": len(
                    [1 for i in range(len(labels)) for j in range(i + 1, len(labels)) if abs(labels[i] - labels[j]) >= 0.75]
                ),
                "valid_pair_count_delta_0_5": len(
                    [1 for i in range(len(labels)) for j in range(i + 1, len(labels)) if abs(labels[i] - labels[j]) >= 0.5]
                ),
            }
        )
    atomic_frame(output / "pairs/per_video_pair_statistics.tsv", statistics)
    chosen = next(row for row in coverage if row["delta"] == selected)
    lines = [
        "# Stage 21A pair coverage",
        "",
        "- Selected delta: **{}**".format(selected),
        "- Train sample coverage: **{:.2%}** ({}/{})".format(
            chosen["sample_coverage_fraction"], chosen["sample_coverage_count"], len(table["ids"])
        ),
        "- Eligible multi-clip source coverage: **{:.2%}** ({}/{})".format(
            chosen["eligible_video_fraction"], chosen["eligible_video_count"], chosen["multi_clip_video_count"]
        ),
        "- Raw qualifying pairs: **{}**".format(chosen["raw_pair_count"]),
        "- Deterministically retained balanced pairs: **{}**".format(len(pairs)),
        "- Each retained source has total raw pair weight 1; at most 64 pairs/source.",
    ]
    atomic_text(output / "pairs/pair_coverage.md", "\n".join(lines) + "\n")
    return report, pairs


class PrimaryCapture:
    def __init__(self, backbone):
        self.value = None
        self.handle = backbone.proj1.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        del module
        self.value = inputs[0].detach()

    def close(self):
        self.handle.remove()


def frozen_cache(cli, args, datasets, tables, output):
    pointer = json.loads(Path(cli.uniform_pointer).read_text())
    run_manifest = json.loads(Path(cli.uniform_run_manifest).read_text())
    checkpoint_sha = sha256_file(cli.uniform_checkpoint)
    if checkpoint_sha != pointer["sha256"] or int(pointer["epoch"]) != 2:
        raise RuntimeError("Uniform selected checkpoint pointer mismatch.")
    args.seq_lens = datasets["train"].get_seq_len()
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    model.load_state_dict(torch.load(cli.uniform_checkpoint, map_location=args.device), strict=True)
    model.eval()
    capture = PrimaryCapture(backbone)
    manifest = {
        "created_at": utc_now(),
        "code_commit": git_head(),
        "checkpoint_path": str(cli.uniform_checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "selected_epoch": int(pointer["epoch"]),
        "selected_valid_J": float(pointer["J"]),
        "batch_size": int(run_manifest["batch_size"]),
        "optimizer_protocol": "Adam lr=1e-4, Stage19 FP32 Uniform KD",
        "missing_schedule": "seed1111+104729, LA/LV/L independent per sample during training",
        "teacher_checkpoint_sha256": json.loads(Path(cli.teacher_cache_manifest).read_text())["entries"]["train"]["teacher_checkpoint_sha256"],
        "feature_name": "actual tensor entering backbone.proj1 after the ModDrop mask residual",
        "secondary_feature_name": "concatenated c_l_sim/c_v_sim/c_a_sim",
        "dtype": "float32",
        "modes": list(MODES),
        "splits": {},
        "locked_test_access_count": 0,
    }
    for split in ALLOWED_SPLITS:
        require_allowed_split(split)
        loader = DataLoader(
            datasets[split],
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(cli.num_workers),
        )
        collected = {mode: {"prediction": [], "primary": [], "secondary": []} for mode in MODES}
        observed_ids = []
        with torch.no_grad():
            for batch in loader:
                text = batch["text"].to(args.device)
                audio = batch["audio"].to(args.device)
                vision = batch["vision"].to(args.device)
                observed_ids.extend(canonical_ids(batch["id"]))
                for mode in MODES:
                    mask = mode_to_mask(mode, len(batch["id"]), args.device, audio.dtype)
                    result = model(text, audio, vision, mask)
                    if capture.value is None:
                        raise RuntimeError("Primary representation hook did not fire.")
                    secondary = torch.cat(
                        [result["c_l_sim"], result["c_v_sim"], result["c_a_sim"]], dim=1
                    )
                    collected[mode]["prediction"].append(result["output_logit"].float().cpu().numpy())
                    collected[mode]["primary"].append(capture.value.float().cpu().numpy())
                    collected[mode]["secondary"].append(secondary.float().cpu().numpy())
        if observed_ids != tables[split]["ids"].tolist():
            raise RuntimeError("Frozen cache sample order mismatch for {}.".format(split))
        arrays = {
            "sample_id": tables[split]["ids"].astype(str),
            "video_id": tables[split]["video_ids"].astype(str),
            "clip_id": tables[split]["clip_ids"].astype(str),
            "label": tables[split]["labels"].astype(np.float32),
        }
        for mode in MODES:
            arrays["prediction_{}".format(mode)] = np.concatenate(collected[mode]["prediction"]).reshape(-1).astype(np.float32)
            arrays["primary_{}".format(mode)] = np.concatenate(collected[mode]["primary"]).astype(np.float32)
            arrays["secondary_{}".format(mode)] = np.concatenate(collected[mode]["secondary"]).astype(np.float32)
        cache_path = output / "representations/{}_cache.npz".format(split)
        atomic_npz(cache_path, **arrays)
        manifest["splits"][split] = {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "sample_count": len(observed_ids),
            "sample_order_sha256": ordered_id_sha(observed_ids),
            "primary_shape": list(arrays["primary_LAV"].shape),
            "secondary_shape": list(arrays["secondary_LAV"].shape),
        }
    capture.close()
    atomic_json(output / "representations/representation_cache_manifest.json", manifest)
    return manifest


def load_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def residual_audit(cli, tables, output):
    report = {
        "created_at": utc_now(),
        "permutation_count": int(cli.permutations),
        "bootstrap_count": int(cli.permutations),
        "splits": {},
        "locked_test_access_count": 0,
    }
    permutation_rows = []
    bootstrap_rows = []
    for split in ALLOWED_SPLITS:
        cache = load_cache(output / "representations/{}_cache.npz".format(split))
        report["splits"][split] = {}
        for mode_position, mode in enumerate(MODES):
            structure, source_rows, residuals = residual_structure(
                tables[split]["video_ids"], tables[split]["labels"], cache["prediction_{}".format(mode)]
            )
            permuted, bootstrap = icc_permutation_and_bootstrap(
                tables[split]["video_ids"],
                residuals,
                permutations=cli.permutations,
                seed=21030 + 100 * mode_position + (0 if split == "train" else 1),
            )
            groups = group_indices(tables[split]["video_ids"])
            remove_count = max(1, int(math.ceil(0.05 * len(groups))))
            largest = set(sorted(groups, key=lambda key: len(groups[key]), reverse=True)[:remove_count])
            worst = set(
                row["video_id"]
                for row in sorted(source_rows, key=lambda row: row["mean_absolute_residual"], reverse=True)[:remove_count]
            )
            kept_largest = [residuals[groups[key]] for key in sorted(groups) if key not in largest]
            kept_worst = [residuals[groups[key]] for key in sorted(groups) if key not in worst]
            structure.update(
                {
                    "shuffled_icc_mean": float(permuted.mean()),
                    "shuffled_icc_p95": float(np.quantile(permuted, 0.95)),
                    "real_icc_percentile": float(np.mean(permuted < structure["icc"])),
                    "permutation_p_value": float((1 + np.sum(permuted >= structure["icc"])) / (len(permuted) + 1)),
                    "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "icc_without_largest_5pct_sources": one_way_icc_from_groups(kept_largest)["icc"],
                    "icc_without_worst_5pct_sources": one_way_icc_from_groups(kept_worst)["icc"],
                }
            )
            structure["gate_checks"] = {
                "icc_ge_0_05": structure["icc"] >= 0.05,
                "above_shuffled_p95": structure["icc"] > structure["shuffled_icc_p95"],
                "direction_survives_largest_sources": structure["icc_without_largest_5pct_sources"] > 0,
                "bootstrap_ci_excludes_zero_positive": structure["bootstrap_ci95_low"] > 0,
            }
            structure["gate_passed"] = all(structure["gate_checks"].values())
            report["splits"][split][mode] = structure
            for index, value in enumerate(permuted):
                permutation_rows.append({"split": split, "mode": mode, "iteration": index, "icc": float(value)})
            for index, value in enumerate(bootstrap):
                bootstrap_rows.append({"split": split, "mode": mode, "iteration": index, "icc": float(value)})
    passed_modes = [mode for mode in MODES if report["splits"]["train"][mode]["gate_passed"]]
    report["train_gate_passed_modes"] = passed_modes
    report["source_residual_gate_passed"] = len(passed_modes) >= 2
    report["status"] = "PASS" if report["source_residual_gate_passed"] else "STAGE21A_FAILED_NO_SOURCE_RESIDUAL_STRUCTURE"
    atomic_json(output / "residual/source_residual_icc.json", report)
    atomic_frame(output / "residual/permutation_icc.tsv", permutation_rows)
    atomic_frame(output / "residual/cluster_bootstrap.tsv", bootstrap_rows)
    lines = [
        "# Stage 21A source residual ICC audit",
        "",
        "- Train-only residual gate: **{}**".format(report["source_residual_gate_passed"]),
        "- Passing Train modes: {}".format(", ".join(passed_modes) if passed_modes else "none"),
        "- 2,000 group-size-preserving permutations and 2,000 source-cluster bootstraps per split/mode.",
        "",
        "| Split | Mode | ICC | shuffled p95 | percentile | p-value | bootstrap 95% CI | without largest 5% | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for split in ALLOWED_SPLITS:
        for mode in MODES:
            value = report["splits"][split][mode]
            lines.append(
                "| {} | {} | {:.6f} | {:.6f} | {:.3f} | {:.5f} | [{:.6f}, {:.6f}] | {:.6f} | {} |".format(
                    split,
                    mode,
                    value["icc"],
                    value["shuffled_icc_p95"],
                    value["real_icc_percentile"],
                    value["permutation_p_value"],
                    value["bootstrap_ci95_low"],
                    value["bootstrap_ci95_high"],
                    value["icc_without_largest_5pct_sources"],
                    value["gate_passed"],
                )
            )
    atomic_text(output / "residual/source_residual_icc.md", "\n".join(lines) + "\n")
    return report


def split_manifests(table, output):
    results = {}
    for seed in (2101, 2102):
        split = make_source_split(table["video_ids"], table["labels"], seed)
        train_index = split.pop("inner_train_indices")
        valid_index = split.pop("inner_valid_indices")
        train_sources = split["inner_train_sources"]
        valid_sources = split["inner_valid_sources"]
        groups = group_indices(table["video_ids"])
        manifest = {
            **split,
            "inner_train_sample_ids": table["ids"][train_index].astype(str).tolist(),
            "inner_valid_sample_ids": table["ids"][valid_index].astype(str).tolist(),
            "inner_train_sample_order_sha256": ordered_id_sha(table["ids"][train_index]),
            "inner_valid_sample_order_sha256": ordered_id_sha(table["ids"][valid_index]),
            "inner_train_label_histogram": label_histogram(table["labels"][train_index]),
            "inner_valid_label_histogram": label_histogram(table["labels"][valid_index]),
            "inner_train_group_size_histogram": label_histogram(np.clip([len(groups[key]) for key in train_sources], -3, 3)),
            "inner_valid_group_size_histogram": label_histogram(np.clip([len(groups[key]) for key in valid_sources], -3, 3)),
            "source_overlap_count": 0,
            "manifest_sha256": None,
            "locked_test_access_count": 0,
        }
        manifest["manifest_sha256"] = sha256_json({key: value for key, value in manifest.items() if key != "manifest_sha256"})
        atomic_json(output / "splits/split_{}_manifest.json".format(seed), manifest)
        results[seed] = manifest
    return results


def main():
    cli = parse_args()
    setup_seed(2100)
    output = Path(cli.output_root)
    output.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)
    datasets = {split: locked_dataset(args, split) for split in ALLOWED_SPLITS}
    tables = {split: dataset_table(datasets[split]) for split in ALLOWED_SPLITS}
    binding = data_binding(cli, datasets, tables, output)
    pair_report, pairs = pair_audit(tables["train"], output)
    if not pairs:
        print(json.dumps({"status": pair_report["status"]}, indent=2))
        return
    cache_manifest = frozen_cache(cli, args, datasets, tables, output)
    residual = residual_audit(cli, tables, output)
    split_manifests(tables["train"], output)
    atomic_json(
        output / "protocol/phase_prepare_summary.json",
        {
            "data_binding_status": binding["status"],
            "selected_delta": pair_report["selected_delta"],
            "selected_pair_count": pair_report["selected_pair_count"],
            "representation_cache_manifest_sha256": sha256_file(output / "representations/representation_cache_manifest.json"),
            "source_residual_gate_passed": residual["source_residual_gate_passed"],
            "code_commit": git_head(),
            "gpu_id": cli.gpu_id,
            "locked_test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "selected_delta": pair_report["selected_delta"],
                "selected_pairs": pair_report["selected_pair_count"],
                "source_residual_gate_passed": residual["source_residual_gate_passed"],
                "cache_splits": list(cache_manifest["splits"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
