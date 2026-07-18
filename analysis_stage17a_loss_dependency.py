"""Static audit of the actual CFCompatKD training objective used by Stage 8."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from trains.singleTask.loss_dependency_registry import (
    LOSS_DEPENDENCIES,
    actual_view_active,
    semantic_active,
)


ROOT = Path("result/missing_baseline/mcao_v1/mosi")
AUDIT = ROOT / "stage17a_loss_audit"
SOURCE_FILES = (
    Path("run_cfcompat_stability_multiseed.py"),
    Path("train_cf_compat_kd.py"),
    Path("trains/singleTask/missing_utils.py"),
    Path("trains/singleTask/model/DLF.py"),
    Path("trains/singleTask/cf_compat_kd_utils.py"),
)
MOSEI_STATE = Path(
    "/code/DLF-mosei-generalization-v1/runtime/mosei_generalization_v1/state.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_background():
    processes = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,stat=,args="], text=True
    ).splitlines()
    processes = [
        line for line in processes
        if "mosei" in line.lower() or "stage10" in line.lower()
    ]
    state = json.loads(MOSEI_STATE.read_text()) if MOSEI_STATE.is_file() else None
    payload = {
        "ReadOnlyCheck": True,
        "Worktrees": subprocess.check_output(
            ["git", "worktree", "list"], text=True
        ).splitlines(),
        "Processes": processes,
        "State": state,
        "MOSEIResultsReadForMethodDevelopment": False,
    }
    (ROOT / "mosei_background_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def dependency_frame():
    rows = []
    for entry in LOSS_DEPENDENCIES:
        row = {
            "LossName": entry.name,
            "ActualView": entry.view,
            "Family": entry.family,
            "OriginalCoefficient": entry.coefficient,
            "RequiredModalities": ",".join(entry.required_modalities) or "always",
            "AlwaysOnByPreregisteredRule": entry.always_on,
            "Formula": entry.formula,
            "DependentTensors": entry.tensors,
            "ParameterGradientPath": entry.gradient_path,
            "CurrentCodeHasPresenceMask": entry.currently_masked,
            "CurrentMaskBasis": "none",
        }
        for mode in ("LAV", "LA", "LV", "L"):
            row["SemanticValid_{}".format(mode)] = semantic_active(entry, mode)
            row["ActuallyComputedInView_{}".format(mode)] = actual_view_active(
                entry, mode
            )
            row["InvalidButComputed_{}".format(mode)] = (
                actual_view_active(entry, mode)
                and not semantic_active(entry, mode)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    record_background()
    frame = dependency_frame()
    frame.to_csv(AUDIT / "stage17_loss_dependency_graph.csv", index=False)
    source_hashes = {str(path): sha256(path) for path in SOURCE_FILES}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    actual_path = [
        "# MCAO Actual CFCompatKD Training Path Specification",
        "",
        "- Audited commit: `{}`".format(commit),
        "- Formal checkpoint producer: `run_cfcompat_stability_multiseed.py::train_one_seed`",
        "- Underlying Stage3 entrypoint: `train_cf_compat_kd.py::train_one_seed`",
        "- Model: `MissingModalityWrapper(DLF)`",
        "- Missing-mode source: `sample_missing_masks` with a dedicated generator seeded by `seed + 104729`",
        "- Full-view loss: `compute_full_dlf_loss(student(..., LAV))`",
        "- Missing-view loss: `compute_task_loss(student(..., sampled_mask))`",
        "- KD: `gated_kd_loss` using the locked train-only compatibility cache",
        "- Backward: `total_loss = full_loss + missing_loss + kd_loss; total_loss.backward()`",
        "- Update: Adam with original gradient accumulation; Stage8 also updates EMA after each optimizer step",
        "- Original DLF trainer inherited: no; the objective and loop are explicitly rebuilt",
        "- Existing auxiliary presence mask: no",
        "- Existing mode-specific handling: input missing tokens/mask residual and KD compatibility only",
        "- Hidden scaling: task weights 1:1:3:1:1; reconstruction/consistency 0.1; orthogonality/similarity 0.01",
        "- Full and sampled-missing views are jointly computed in every training batch",
        "- Historical Stage8 loop constructs a test loader and evaluates test each epoch: yes",
        "- Stage17 may not reuse that evaluation loop; Stage17A constructs train/valid surfaces only",
        "",
        "## Exact call chain",
        "",
        "`main -> train_one_seed -> initialize_teacher_student -> "
        "compute_full_dlf_loss + compute_task_loss + gated_kd_loss -> "
        "total_loss.backward -> optimizer_step_and_update_ema`",
        "",
        "## Source SHA-256",
        "",
    ]
    actual_path.extend(
        "- `{}`: `{}`".format(path, digest)
        for path, digest in source_hashes.items()
    )
    Path("MCAO_ACTUAL_TRAINING_PATH_SPEC.md").write_text(
        "\n".join(actual_path) + "\n"
    )

    invalid = frame.loc[
        frame[
            [
                "InvalidButComputed_LA",
                "InvalidButComputed_LV",
                "InvalidButComputed_L",
            ]
        ].any(axis=1)
    ]
    dependency_doc = [
        "# MCAO Loss Dependency Specification",
        "",
        "The actual CFCompatKD objective has two student views per batch. The full "
        "LAV view receives the complete DLF objective. The sampled missing view "
        "receives five task heads plus compatibility-gated prediction KD.",
        "",
        "The sampled missing-view task helper has no presence argument. Therefore "
        "`missing_audio_specific_task` and `missing_visual_specific_task` remain "
        "in the scalar total even when their source modality is absent. Missing-"
        "view reconstruction, consistency, orthogonality, and triplet losses are "
        "not computed at all; their full-LAV counterparts remain semantically valid.",
        "",
        "## Statically inconsistent losses",
        "",
    ]
    dependency_doc.extend(
        "- `{}`: invalid-but-computed modes {}".format(
            row.LossName,
            ", ".join(
                mode
                for mode in ("LA", "LV", "L")
                if getattr(row, "InvalidButComputed_{}".format(mode))
            ),
        )
        for row in invalid.itertuples()
    )
    dependency_doc.extend(
        [
            "",
            "The CSV is the authoritative row-level dependency graph:",
            "`result/missing_baseline/mcao_v1/mosi/stage17a_loss_audit/"
            "stage17_loss_dependency_graph.csv`.",
            "",
        ]
    )
    Path("MCAO_LOSS_DEPENDENCY_SPEC.md").write_text(
        "\n".join(dependency_doc)
    )

    fair = [
        "# MCAO Fair Training Path Specification",
        "",
        "- Actual student initialization: a `MissingModalityWrapper` whose DLF "
        "backbone is loaded from the matched clean seed checkpoint.",
        "- The CFCompatKD path does not initialize from a separately trained "
        "ModDrop checkpoint; no ModDrop retraining may be added.",
        "- Original, MCAO-PM, and MCAO-AN must use the same clean teacher/student "
        "initial state, locked compatibility cache, missing sequence, data order, "
        "RNG, optimizer, scheduler, accumulation, clipping, and epoch rule.",
        "- Stage17 must replace the historical per-epoch valid/test loop with the "
        "preregistered video-group inner split and a single official-valid evaluation.",
        "- Official valid and test cannot select PM versus AN. The formal variant "
        "is frozen from Stage17A train-only expected-contribution imbalance.",
        "",
    ]
    Path("MCAO_FAIR_TRAINING_PATH_SPEC.md").write_text("\n".join(fair))

    manifest = {
        "Commit": commit,
        "ActualEntrypoint": "run_cfcompat_stability_multiseed.py::train_one_seed",
        "SourceSHA256": source_hashes,
        "ActualLossCount": len(frame),
        "StaticInvalidButComputedLosses": invalid.LossName.tolist(),
        "PresenceMaskAlreadyImplemented": False,
        "HistoricalLoopEvaluatesTestEachEpoch": True,
        "TestLoaderConstructed": False,
        "TestFeaturesRead": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
    }
    (AUDIT / "static_training_path_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(frame[[
        "LossName", "OriginalCoefficient", "RequiredModalities",
        "InvalidButComputed_LA", "InvalidButComputed_LV", "InvalidButComputed_L",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
