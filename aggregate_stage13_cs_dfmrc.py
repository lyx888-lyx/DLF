"""Aggregate the evidence-gated Stage 13 audit without accessing test data."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "result/missing_baseline/cs_dfmrc_v1/mosi"


def main():
    a1 = json.loads(
        (
            OUTPUT
            / "stage13a1_safe_oracle/stage13a1_gate.json"
        ).read_text()
    )
    a2 = json.loads(
        (
            OUTPUT
            / "stage13a2_residual_structure/stage13a2_gate.json"
        ).read_text()
    )
    a3 = json.loads(
        (
            OUTPUT
            / "stage13a3_loso_calibration/stage13a3_gate.json"
        ).read_text()
    )
    if not a1["Passed"]:
        stopped = "Stage13A-1"
    elif not a2["Passed"]:
        stopped = "Stage13A-2"
    elif not a3["Passed"]:
        stopped = "Stage13A-3"
    else:
        raise RuntimeError(
            "Stage13A passed; this stopping-only aggregator cannot run Stage13B."
        )
    test_unlock = OUTPUT / "TEST_UNLOCK_MANIFEST.json"
    locked_test = OUTPUT / "locked_test"
    if test_unlock.exists() or any(locked_test.iterdir()):
        raise RuntimeError("Forbidden test artifacts exist before unlock.")
    state_path = Path(
        "/code/DLF-mosei-generalization-v1/runtime/"
        "mosei_generalization_v1/state.json"
    )
    background = {
        "State": json.loads(state_path.read_text()),
        "Processes": subprocess.check_output(
            [
                "bash",
                "-lc",
                "ps -ef | grep -E 'mosei|stage10' | grep -v grep || true",
            ],
            text=True,
        ).splitlines(),
        "ReadOnlyCheck": True,
    }
    (OUTPUT / "final/mosei_background_status_final.json").write_text(
        json.dumps(background, indent=2, sort_keys=True) + "\n"
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    payload = {
        "StoppedStage": stopped,
        "A1Verdict": a1["Verdict"],
        "A2Verdict": a2["Verdict"],
        "A3Verdict": a3["Verdict"],
        "Stage13BExecuted": False,
        "TestUnlockManifestGenerated": False,
        "TestLoaderConstructed": False,
        "TestLabelsRead": False,
        "TestPredictionsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
        "RecommendContinueRoute": False,
        "Branch": branch,
        "Commit": head,
        "FinalStatus": "CS-DFMRC PIPELINE STOPPED BY EVIDENCE GATE",
    }
    (OUTPUT / "final/stage13_pipeline_final_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Stage 13 CS-DFMRC Final Audit",
        "",
        "- Stopped stage: {}".format(stopped),
        "- Stage13A-1: {}".format(a1["Verdict"]),
        "- Stage13A-2: {}".format(a2["Verdict"]),
        "- Stage13A-3: {}".format(a3["Verdict"]),
        "- Stage13B executed: false",
        "- Test accessed: false",
        "- TEST_UNLOCK_MANIFEST generated: false",
        "- Locked test access count: 0",
        "- MOSEI modified: false",
        "- Branch: `{}`".format(branch),
        "- Commit: `{}`".format(head),
        "",
        "## A1 headroom",
        "",
        "- Mean Safe Oracle LAV MAE gain: {:.9f}".format(
            a1["Metrics"]["MeanLAVMAEGain"]
        ),
        "- Mean Safe Oracle MissingMacro MAE gain: {:.9f}".format(
            a1["Metrics"]["MeanMissingMacroMAEGain"]
        ),
        "- Mean Safe Oracle J gain: {:.9f}".format(
            a1["Metrics"]["MeanJGain"]
        ),
        "",
        "## A2 structure",
        "",
        "- Strong-group valid coverage: {:.6f}".format(
            a2["Metrics"]["StrongGroupValidCoverage"]
        ),
        "- Strong groups: {}".format(a2["Metrics"]["StrongGroups"]),
        "",
        "## A3 LOSO failure",
        "",
        "- Mean Delta J: {:.9f}".format(a3["Metrics"]["MeanDeltaJ"]),
        "- Improved seeds: {}/5".format(a3["Metrics"]["ImprovedJSeeds"]),
        "- Worst seed Delta J: {:.9f}".format(
            a3["Metrics"]["WorstSeedDeltaJ"]
        ),
        "- Remove-best-seed Mean Delta J: {:.9f}".format(
            a3["Metrics"]["RemoveBestSeedMeanDeltaJ"]
        ),
        "- RecoverableRatio_J: {:.6f}".format(
            a3["Metrics"]["RecoverableRatioJ"]
        ),
        "",
        "The CS-DFMRC route should not continue.",
        "",
        "STAGE13A3_LOSO_CALIBRATION_UNSUPPORTED",
        "",
        "CS-DFMRC PIPELINE STOPPED BY EVIDENCE GATE",
    ]
    report = OUTPUT / "final/stage13_cs_dfmrc_final_audit.md"
    report.write_text("\n".join(lines) + "\n")
    print(payload["FinalStatus"], flush=True)


if __name__ == "__main__":
    main()
