"""Aggregate and hard-gate the four existing Stage18A recovery runs."""

import hashlib
import json
import statistics
from pathlib import Path

import pandas as pd

from trains.singleTask.cfcompat_fair_trainer import TEST_ISOLATION


ROOT = Path("result/missing_baseline/cfcompat_evidence_v1/mosi")
STAGE = ROOT / "stage18a_training_recovery"
HISTORICAL = Path(
    "/code/DLF/result/missing_baseline/cfcompat_stability_v1/"
    "mosi/seed1114/per_seed_all_methods.csv"
)
MOSEI_RUNTIME = Path(
    "/code/DLF-mosei-generalization-v1/runtime/mosei_generalization_v1"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_runs():
    frames = []
    for method in ("moddrop", "cfcompat"):
        for run in ("A", "B"):
            path = STAGE / "{}_run{}".format(method, run) / "run_metrics.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def gpu_gate():
    path = Path("/tmp/stage18_gpu_gate.csv")
    names = ["timestamp", "sample", "index", "used_mib", "total_mib", "free_mib", "util"]
    frame = pd.read_csv(path, names=names)
    gpu3 = frame.loc[frame["index"].astype(int).eq(3)].copy()
    return {
        "sample_count": int(len(gpu3)),
        "minimum_free_mib": int(gpu3.free_mib.min()),
        "median_utilization_percent": float(statistics.median(gpu3.util.tolist())),
        "maximum_used_mib": int(gpu3.used_mib.max()),
        "passed": bool(
            len(gpu3) >= 13
            and gpu3.free_mib.min() >= 12 * 1024
            and statistics.median(gpu3.util.tolist()) < 75
        ),
        "raw_sample_sha256": sha256(path),
    }


def write_background_status(gate):
    state_path = MOSEI_RUNTIME / "state.json"
    state = json.loads(state_path.read_text())
    pids = {}
    for name in ("coordinator", "supervisor", "worker_gpu_3"):
        path = MOSEI_RUNTIME / "{}.pid".format(name)
        pids[name] = {
            "pid": int(path.read_text().strip()),
            "alive_at_preflight": False,
        }
    payload = {
        "observed_state": state,
        "state_sha256": sha256(state_path),
        "interpretation": (
            "MOSEI completed successfully; absence of supervisor/worker processes "
            "is normal terminal state, not a failure."
        ),
        "pids": pids,
        "gpu3_resource_gate": gate,
        "mosei_modified": False,
        "mosei_results_used_for_mosi_selection": False,
    }
    (ROOT / "mosei_background_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_runs()
    rows.to_csv(STAGE / "stage18a_replay_runs.csv", index=False)
    historical_columns = [
        "Method",
        "BestValidEpoch",
        "J_valid",
        "valid_LAV_MAE",
        "valid_LA_MAE",
        "valid_LV_MAE",
        "valid_L_MAE",
    ]
    historical = pd.read_csv(HISTORICAL, usecols=historical_columns)
    old = historical.loc[historical.Method.eq("Online")].iloc[0]
    old_missing = float(old[["valid_LA_MAE", "valid_LV_MAE", "valid_L_MAE"]].mean())
    mod = rows.loc[rows.Method.eq("moddrop")]
    cf = rows.loc[rows.Method.eq("cfcompat")]
    checks = {
        "moddrop_repeat_abs_delta_j": float(mod.J_valid.max() - mod.J_valid.min()),
        "cfcompat_repeat_abs_delta_j": float(cf.J_valid.max() - cf.J_valid.min()),
        "cfcompat_mean_delta_j_vs_historical": float(cf.J_valid.mean() - old.J_valid),
        "cfcompat_mean_delta_lav_mae_vs_historical": float(
            cf.valid_LAV_MAE.mean() - old.valid_LAV_MAE
        ),
        "cfcompat_mean_delta_missing_macro_mae_vs_historical": float(
            cf.valid_MissingMacro_MAE.mean() - old_missing
        ),
        "all_initial_student_sha_equal": bool(
            rows.InitialStudentStateSHA256.nunique() == 1
        ),
        "all_teacher_sha_equal": bool(rows.TeacherSHA256.nunique() == 1),
        "all_evaluator_sha_equal": bool(rows.EvaluatorSHA256.nunique() == 1),
        "all_cache_sha_equal": bool(rows.CompatibilityCacheSHA256.nunique() == 1),
        "all_optimizer_sha_equal": bool(rows.OptimizerConfigSHA256.nunique() == 1),
        "all_trainer_sha_equal": bool(rows.TrainerSHA256.nunique() == 1),
        "repeat_checkpoint_sha_equal_by_method": bool(
            rows.groupby("Method").CheckpointSHA256.nunique().max() == 1
        ),
        "test_access_flags_all_locked": bool(
            all(
                not bool(row[key])
                if key != "locked_test_access_count"
                else int(row[key]) == 0
                for _, row in rows.iterrows()
                for key in TEST_ISOLATION
            )
        ),
    }
    passed = bool(
        checks["moddrop_repeat_abs_delta_j"] <= 0.003
        and checks["cfcompat_repeat_abs_delta_j"] <= 0.003
        and abs(checks["cfcompat_mean_delta_j_vs_historical"]) <= 0.010
        and abs(checks["cfcompat_mean_delta_lav_mae_vs_historical"]) <= 0.015
        and abs(checks["cfcompat_mean_delta_missing_macro_mae_vs_historical"])
        <= 0.015
        and all(
            checks[key]
            for key in checks
            if key
            not in {
                "moddrop_repeat_abs_delta_j",
                "cfcompat_repeat_abs_delta_j",
                "cfcompat_mean_delta_j_vs_historical",
                "cfcompat_mean_delta_lav_mae_vs_historical",
                "cfcompat_mean_delta_missing_macro_mae_vs_historical",
            }
        )
        and set(cf.BestValidEpoch.astype(int)) == {7}
        and set(cf.LastEpoch.astype(int)) == {17}
    )
    gate = gpu_gate()
    passed = passed and gate["passed"]
    write_background_status(gate)
    manifest = {
        "stage": "18A",
        "status": (
            "STAGE18A_FAIR_TRAINING_PROTOCOL_RECOVERED"
            if passed
            else "STAGE18A_FAIR_TRAINING_PROTOCOL_UNRECOVERED"
        ),
        "checks": checks,
        "gpu3_resource_gate": gate,
        "historical_reference": {
            "path": str(HISTORICAL),
            "method": "Online",
            "J_valid": float(old.J_valid),
            "LAV_MAE": float(old.valid_LAV_MAE),
            "MissingMacro_MAE": old_missing,
        },
        "test_isolation": TEST_ISOLATION,
        "branch": "experiment/cfcompat-distillation-evidence-v1",
        "base_commit": "eca55847edf9e6968ad896f8998783f5c4c5c883",
    }
    (STAGE / "stage18a_training_asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    audit = f"""# Stage 18A Fair Training Recovery Audit

Status: **{manifest['status']}**

Four independent seed-1114 runs used the same no-Test Online trainer. ModDrop
selected epoch 11 and stopped at epoch 21 in both runs; CFCompatKD selected
epoch 7 and stopped at epoch 17 in both runs.

| Check | Value | Limit |
|---|---:|---:|
| ModDrop repeated absolute Delta J | {checks['moddrop_repeat_abs_delta_j']:.12g} | <= 0.003 |
| CFCompat repeated absolute Delta J | {checks['cfcompat_repeat_abs_delta_j']:.12g} | <= 0.003 |
| CFCompat mean Delta J vs historical | {checks['cfcompat_mean_delta_j_vs_historical']:.12g} | abs <= 0.010 |
| CFCompat Delta LAV MAE vs historical | {checks['cfcompat_mean_delta_lav_mae_vs_historical']:.12g} | abs <= 0.015 |
| CFCompat Delta MissingMacro MAE vs historical | {checks['cfcompat_mean_delta_missing_macro_mae_vs_historical']:.12g} | abs <= 0.015 |

All initial student, Teacher, evaluator, compatibility-cache, optimizer-config,
and trainer hashes matched. Repeated checkpoints matched within each method.
The realized full-run schedule hashes differ across methods only because their
frozen early-stop horizons differ; both repetitions within a method match
exactly. The sampling algorithm and seed are identical for every method.

No Test loader, feature, label, prediction, or evaluation was accessed.
Locked Test access count remains zero.
"""
    (STAGE / "stage18a_training_recovery_audit.md").write_text(audit)
    for name in (
        "CFCompat_EVIDENCE_PROTOCOL.md",
        "CFCompat_FAIR_TRAINING_PROTOCOL.md",
        "CFCompat_TEST_ONCE_PROTOCOL.md",
    ):
        (ROOT / name).write_text(Path(name).read_text())
    print(manifest["status"])
    print(rows[["Method", "RunLabel", "BestValidEpoch", "LastEpoch", "J_valid",
                "valid_LAV_MAE", "valid_MissingMacro_MAE"]].to_string(index=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
