"""CPU-only, train/valid-only Stage 15A MOSI context feasibility audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.causal_context_index import (
    CONTEXT_LENGTH,
    build_context_index,
    canonical_context_sha,
    deterministic_hash_choice,
    parse_split,
    validate_context_index,
)


RESULT_ROOT = Path("result/missing_baseline/mcdc_v1/mosi")
AUDIT_ROOT = RESULT_ROOT / "stage15a_context_audit"
TRAIN_SOURCE = Path(
    "/code/DLF/result/counterfactual_compatibility/"
    "cf_compat_v1/mosi/train_counterfactual_compatibility.csv"
)
VALID_SOURCE = Path(
    "/code/DLF/result/missing_baseline/cfcompat_prediction_ensemble_v1/"
    "mosi/online_seed1114_valid_predictions.csv"
)
MOSEI_STATE = Path(
    "/code/DLF-mosei-generalization-v1/runtime/mosei_generalization_v1/state.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command):
    return subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ).stdout


def record_mosei_status(path: Path) -> None:
    state = None
    if MOSEI_STATE.is_file():
        state = json.loads(MOSEI_STATE.read_text())
    payload = {
        "ReadOnlyCheck": True,
        "Worktrees": command_output(["git", "worktree", "list"]).splitlines(),
        "Processes": command_output(["ps", "-ef"]).splitlines(),
        "NvidiaSMI": command_output(["nvidia-smi"]),
        "State": state,
    }
    payload["Processes"] = [
        line for line in payload["Processes"] if "mosei" in line.lower() or "stage10" in line.lower()
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def coverage_rows(context_by_split):
    rows = []
    for split, frame in context_by_split.items():
        count = len(frame)
        lengths = frame.context_length.to_numpy(dtype=int)
        groups = frame.groupby("video_id").segment_index
        missing_slots = 0
        total_slots = 0
        gap_count = 0
        adjacency_count = 0
        for _, values in groups:
            ordered = np.sort(values.to_numpy(dtype=int))
            if len(ordered):
                total_slots += int(ordered[-1] - ordered[0] + 1)
                missing_slots += int(ordered[-1] - ordered[0] + 1 - len(ordered))
            if len(ordered) > 1:
                gaps = np.diff(ordered)
                gap_count += int(np.sum(gaps != 1))
                adjacency_count += int(len(gaps))
        rows.append(
            {
                "split": split,
                "sample_count": count,
                "video_count": int(frame.video_id.nunique()),
                "history_ge_1_ratio": float(np.mean(lengths >= 1)),
                "history_ge_2_ratio": float(np.mean(lengths >= 2)),
                "history_ge_3_ratio": float(np.mean(lengths >= 3)),
                "first_segment_ratio": float(np.mean(lengths == 0)),
                "missing_number_ratio": float(missing_slots / total_slots) if total_slots else 0.0,
                "nonconsecutive_adjacency_ratio": (
                    float(gap_count / adjacency_count) if adjacency_count else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def video_statistics(parsed_by_split):
    rows = []
    for split, frame in parsed_by_split.items():
        for video_id, local in frame.groupby("video_id", sort=True):
            ordered = np.sort(local.segment_index.to_numpy(dtype=int))
            span = int(ordered[-1] - ordered[0] + 1)
            gaps = int(span - len(ordered))
            rows.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "segment_count": int(len(ordered)),
                    "min_segment_index": int(ordered[0]),
                    "max_segment_index": int(ordered[-1]),
                    "missing_segment_count": gaps,
                    "is_contiguous": bool(gaps == 0 and np.all(np.diff(ordered) == 1)),
                }
            )
    return pd.DataFrame(rows)


def rankdata(values):
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy()


def spearman(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])


def label_continuity(train):
    label_by_id = dict(zip(train.sample_id.astype(str), train.label.astype(float)))
    by_video = {
        str(video): local.sort_values("segment_index", kind="mergesort")
        for video, local in train.groupby("video_id", sort=True)
    }
    rows = []
    for lag in (1, 2, 3):
        left, right = [], []
        for local in by_video.values():
            values = local.label.to_numpy(dtype=float)
            if len(values) > lag:
                left.extend(values[:-lag])
                right.extend(values[lag:])
        rows.append(
            {
                "diagnostic": "lag_{}_label_spearman".format(lag),
                "value": spearman(left, right),
                "pair_count": len(left),
            }
        )
    adjacent_left, adjacent_right = [], []
    same_video_random_left, same_video_random_right = [], []
    cross_video_left, cross_video_right = [], []
    videos = sorted(by_video)
    all_distance, all_difference = [], []
    for video, local in by_video.items():
        labels = local.label.to_numpy(dtype=float)
        ids = local.sample_id.astype(str).tolist()
        if len(labels) > 1:
            adjacent_left.extend(labels[:-1])
            adjacent_right.extend(labels[1:])
        for sample_id, label in zip(ids, labels):
            alternatives = [value for value in ids if value != sample_id]
            if alternatives:
                chosen = deterministic_hash_choice(alternatives, "same|" + sample_id)
                same_video_random_left.append(label)
                same_video_random_right.append(label_by_id[chosen])
            other_video = deterministic_hash_choice(
                [value for value in videos if value != video], "video|" + sample_id
            )
            chosen = deterministic_hash_choice(
                by_video[other_video].sample_id.astype(str).tolist(), "cross|" + sample_id
            )
            cross_video_left.append(label)
            cross_video_right.append(label_by_id[chosen])
        segments = local.segment_index.to_numpy(dtype=int)
        for first in range(len(labels)):
            for second in range(first + 1, len(labels)):
                all_distance.append(int(segments[second] - segments[first]))
                all_difference.append(abs(float(labels[second] - labels[first])))
    adjacent_left = np.asarray(adjacent_left)
    adjacent_right = np.asarray(adjacent_right)
    rows.extend(
        [
            {
                "diagnostic": "adjacent_mean_absolute_label_difference",
                "value": float(np.mean(np.abs(adjacent_left - adjacent_right))),
                "pair_count": len(adjacent_left),
            },
            {
                "diagnostic": "adjacent_same_polarity_ratio",
                "value": float(np.mean((adjacent_left >= 0) == (adjacent_right >= 0))),
                "pair_count": len(adjacent_left),
            },
            {
                "diagnostic": "adjacent_same_acc7_ratio",
                "value": float(np.mean(np.rint(adjacent_left) == np.rint(adjacent_right))),
                "pair_count": len(adjacent_left),
            },
            {
                "diagnostic": "same_video_hash_control_mean_absolute_difference",
                "value": float(
                    np.mean(
                        np.abs(
                            np.asarray(same_video_random_left)
                            - np.asarray(same_video_random_right)
                        )
                    )
                ),
                "pair_count": len(same_video_random_left),
            },
            {
                "diagnostic": "cross_video_hash_control_mean_absolute_difference",
                "value": float(
                    np.mean(np.abs(np.asarray(cross_video_left) - np.asarray(cross_video_right)))
                ),
                "pair_count": len(cross_video_left),
            },
            {
                "diagnostic": "segment_distance_vs_absolute_label_difference_spearman",
                "value": spearman(all_distance, all_difference),
                "pair_count": len(all_distance),
            },
        ]
    )
    return pd.DataFrame(rows)


def write_specs(parsed_by_split):
    examples = []
    for split, frame in parsed_by_split.items():
        examples.extend("{}: `{}`".format(split, value) for value in frame.sample_id.head(3))
    Path("MCDC_SAMPLE_ID_SPEC.md").write_text(
        "# MCDC MOSI Sample ID Specification\n\n"
        "Observed examples: {}.\n\n".format(", ".join(examples))
        + "The exact parser is `^(?P<video_id>.+?)\\\\$_\\\\$(?P<segment_index>[0-9]+)$`. "
        "`video_id` is the non-empty prefix before the final `$_$` delimiter and "
        "`segment_index` is the trailing non-negative integer. IDs that do not match "
        "raise an error. Duplicate `(split, video_id, segment_index)` bindings are fatal. "
        "Context is reconstructed by exact integer offsets `i-3`, `i-2`, `i-1`; missing "
        "offsets remain left padding and are never filled from another sample.\n"
    )


def main():
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    record_mosei_status(RESULT_ROOT / "mosei_background_status.json")
    train_raw = pd.read_csv(TRAIN_SOURCE, usecols=["sample_index", "sample_id", "label"])
    valid_raw = pd.read_csv(VALID_SOURCE, usecols=["sample_index", "sample_id"])
    parsed = {
        "train": parse_split(train_raw, "train"),
        "valid": parse_split(valid_raw, "valid"),
    }
    parsed["train"]["label"] = train_raw.sort_values(
        "sample_index", kind="mergesort"
    ).label.to_numpy(dtype=float)
    video_intersection = set(parsed["train"].video_id) & set(parsed["valid"].video_id)
    context = {split: build_context_index(frame) for split, frame in parsed.items()}
    repeated = {split: build_context_index(frame) for split, frame in parsed.items()}
    context_sha = {split: canonical_context_sha(frame) for split, frame in context.items()}
    repeat_sha = {split: canonical_context_sha(frame) for split, frame in repeated.items()}
    binding_rows = []
    for split in ("train", "valid"):
        errors = validate_context_index(context[split], parsed[split])
        binding_rows.append(
            {
                "split": split,
                **errors,
                "total_binding_errors": int(sum(errors.values())),
                "context_sha256": context_sha[split],
                "repeat_sha256": repeat_sha[split],
                "deterministic": context_sha[split] == repeat_sha[split],
            }
        )
    bindings = pd.DataFrame(binding_rows)
    coverage = coverage_rows(context)
    videos = video_statistics(parsed)
    continuity = label_continuity(parsed["train"])
    id_rows = []
    for split, frame in parsed.items():
        for row in frame.itertuples(index=False):
            id_rows.append(
                {
                    "split": split,
                    "sample_index": int(row.sample_index),
                    "sample_id": row.sample_id,
                    "video_id": row.video_id,
                    "segment_index": int(row.segment_index),
                    "parse_success": True,
                }
            )
    pd.DataFrame(id_rows).to_csv(AUDIT_ROOT / "stage15a_id_parse_audit.csv", index=False)
    videos.to_csv(AUDIT_ROOT / "stage15a_video_statistics.csv", index=False)
    coverage.to_csv(AUDIT_ROOT / "stage15a_context_coverage.csv", index=False)
    continuity.to_csv(AUDIT_ROOT / "stage15a_label_continuity.csv", index=False)
    bindings.to_csv(AUDIT_ROOT / "stage15a_context_binding_verification.csv", index=False)
    context["train"].to_csv(AUDIT_ROOT / "context_index_train.csv", index=False)
    context["valid"].to_csv(AUDIT_ROOT / "context_index_valid.csv", index=False)
    write_specs(parsed)
    gates = {
        "TrainIDParseRate100Percent": len(parsed["train"]) == len(train_raw),
        "ValidIDParseRate100Percent": len(parsed["valid"]) == len(valid_raw),
        "UniqueSplitVideoSegment": all(
            not frame.duplicated(["video_id", "segment_index"]).any() for frame in parsed.values()
        ),
        "NonEmptyVideoID": all(frame.video_id.astype(bool).all() for frame in parsed.values()),
        "TrainValidVideoIntersectionZero": not video_intersection,
        "ContextBindingErrorsZero": bool((bindings.total_binding_errors == 0).all()),
        "DeterministicContextSHA": bool(bindings.deterministic.all()),
        "TrainHistoryGe1AtLeast0.70": float(
            coverage.loc[coverage.split == "train", "history_ge_1_ratio"].iloc[0]
        ) >= 0.70,
        "ValidHistoryGe1AtLeast0.70": float(
            coverage.loc[coverage.split == "valid", "history_ge_1_ratio"].iloc[0]
        ) >= 0.70,
        "TrainHistoryGe2AtLeast0.50": float(
            coverage.loc[coverage.split == "train", "history_ge_2_ratio"].iloc[0]
        ) >= 0.50,
        "ValidHistoryGe2AtLeast0.50": float(
            coverage.loc[coverage.split == "valid", "history_ge_2_ratio"].iloc[0]
        ) >= 0.50,
        "TrainHistoryGe3AtLeast0.35": float(
            coverage.loc[coverage.split == "train", "history_ge_3_ratio"].iloc[0]
        ) >= 0.35,
        "ValidHistoryGe3AtLeast0.35": float(
            coverage.loc[coverage.split == "valid", "history_ge_3_ratio"].iloc[0]
        ) >= 0.35,
    }
    id_gates = [
        "TrainIDParseRate100Percent",
        "ValidIDParseRate100Percent",
        "UniqueSplitVideoSegment",
        "NonEmptyVideoID",
        "TrainValidVideoIntersectionZero",
    ]
    id_pass = all(gates[key] for key in id_gates)
    coverage_pass = all(gates.values())
    verdict = (
        "STAGE15A_MOSI_CONTEXT_STRUCTURE_SUPPORTED"
        if coverage_pass
        else (
            "STAGE15A_SEQUENCE_ID_AUDIT_FAILED"
            if not id_pass
            else "STAGE15A_INSUFFICIENT_CONTEXT_COVERAGE"
        )
    )
    audit = {
        "Verdict": verdict,
        "Passed": coverage_pass,
        "ContextLength": CONTEXT_LENGTH,
        "Gates": gates,
        "TrainSource": str(TRAIN_SOURCE),
        "TrainSourceSHA256": sha256(TRAIN_SOURCE),
        "ValidSource": str(VALID_SOURCE),
        "ValidSourceSHA256": sha256(VALID_SOURCE),
        "ContextSHA256": context_sha,
        "TestLoaderConstructed": False,
        "TestFeaturesRead": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
    }
    (AUDIT_ROOT / "stage15a_gate.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    summary = ["# Stage 15A MOSI Context Feasibility Audit", "", "- Verdict: `{}`".format(verdict)]
    for row in coverage.itertuples(index=False):
        summary.extend(
            [
                "- {}: {} samples, {} videos".format(row.split, row.sample_count, row.video_count),
                "  - history >=1: {:.6%}".format(row.history_ge_1_ratio),
                "  - history >=2: {:.6%}".format(row.history_ge_2_ratio),
                "  - history >=3: {:.6%}".format(row.history_ge_3_ratio),
                "  - first segment: {:.6%}".format(row.first_segment_ratio),
                "  - missing-number ratio: {:.6%}".format(row.missing_number_ratio),
            ]
        )
    summary.extend(
        [
            "- Train/valid video intersection: {}".format(len(video_intersection)),
            "- Context binding errors: {}".format(int(bindings.total_binding_errors.sum())),
            "- Test accessed: false",
            "- Locked test access count: 0",
            "",
            verdict,
            "",
        ]
    )
    (AUDIT_ROOT / "stage15a_context_feasibility_audit.md").write_text("\n".join(summary))
    print(coverage.to_string(index=False))
    print(continuity.to_string(index=False))
    print(verdict)


if __name__ == "__main__":
    main()
