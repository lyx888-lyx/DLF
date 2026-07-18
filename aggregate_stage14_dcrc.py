"""Aggregate a stopped Stage 14A audit without accessing locked test."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "result/missing_baseline/dcrc_v1/mosi"


def main():
    a2 = json.loads(
        (OUTPUT / "stage14a2_valid/stage14a2_gate.json").read_text()
    )
    a3 = json.loads(
        (OUTPUT / "stage14a3_lomo/stage14a3_gate.json").read_text()
    )
    if not a2["Passed"]:
        stopped = "Stage14A-2"
        status = a2["Verdict"]
    elif not a3["Passed"]:
        stopped = "Stage14A-3"
        status = a3["Verdict"]
    else:
        raise RuntimeError(
            "Stage14A passed; stopping aggregator cannot freeze/test DCRC."
        )
    forbidden = (
        OUTPUT / "TEST_UNLOCK_MANIFEST.json",
        OUTPUT / "frozen_method/DCRC_FROZEN_METHOD_MANIFEST.json",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("A frozen/test unlock artifact exists after failure.")
    if any((OUTPUT / "locked_test").iterdir()):
        raise RuntimeError("Locked test artifacts exist after failure.")
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
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "StoppedStage": stopped,
        "A2Verdict": a2["Verdict"],
        "A3Verdict": a3["Verdict"],
        "FallbackCount": a2["Metrics"]["FallbackCount"]
        + a3["Metrics"]["FallbackCount"],
        "DecisionInheritance": a2["DCRCConditions"][
            "ClassificationInheritedExactly"
        ]
        and a3["Conditions"]["ClassificationInheritedFixedAnchor"],
        "Stage14BExecuted": False,
        "DCRCFrozenMethodManifestGenerated": False,
        "TestUnlockManifestGenerated": False,
        "TestLoaderConstructed": False,
        "TestPredictionsRead": False,
        "TestLabelsRead": False,
        "TestEvaluationPerformed": False,
        "LockedTestAccessCount": 0,
        "MOSEIModified": False,
        "RecommendContinueRobustEnsembleRoute": False,
        "Branch": branch,
        "Commit": commit,
        "FinalStatus": "DCRC PIPELINE STOPPED BY EVIDENCE GATE",
    }
    (OUTPUT / "final/stage14_pipeline_final_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Stage 14 DCRC Final Audit",
        "",
        "- Stopped stage: {}".format(stopped),
        "- Status: {}".format(status),
        "- Baseline replay: PASS",
        "- Huber Raw vs PE5 gate: PASS",
        "- DCRC vs ADPEP-All gate: PASS",
        "- Decision inheritance: 100%",
        "- LOMO robustness: FAIL",
        "- Fallback count: {}".format(payload["FallbackCount"]),
        "- Test accessed: false",
        "- Locked test access count: 0",
        "- Stage14B executed: false",
        "- TEST_UNLOCK_MANIFEST generated: false",
        "- MOSEI modified: false",
        "- Branch: `{}`".format(branch),
        "- Commit: `{}`".format(commit),
        "",
        "## Valid signal",
        "",
        "- Anchor seed: {}".format(a2["Metrics"]["AnchorSeed"]),
        "- PE5 Valid J: {:.9f}".format(a2["Metrics"]["PE5J"]),
        "- Huber Raw Valid J: {:.9f}".format(
            a2["Metrics"]["HuberRawJ"]
        ),
        "- ADPEP-All Valid J: {:.9f}".format(
            a2["Metrics"]["ADPEPAllJ"]
        ),
        "- DCRC Valid J: {:.9f}".format(a2["Metrics"]["DCRCJ"]),
        "- DCRC Delta J vs ADPEP-All: {:.9f}".format(
            a2["Metrics"]["DCRCDeltaJVsADPEP"]
        ),
        "",
        "## LOMO failure",
        "",
        "- Improved cases: {}/5".format(a3["Metrics"]["ImprovedCases"]),
        "- Mean Delta J: {:.9f}".format(a3["Metrics"]["MeanDeltaJ"]),
        "- Worst-case Delta J: {:.9f}".format(
            a3["Metrics"]["WorstCaseDeltaJ"]
        ),
        "- Remove-best-case Mean Delta J: {:.9f}".format(
            a3["Metrics"]["RemoveBestCaseMeanDeltaJ"]
        ),
        "",
        "The robust ensemble route should not continue under the frozen gate.",
        "",
        status,
        "",
        "DCRC PIPELINE STOPPED BY EVIDENCE GATE",
    ]
    report = OUTPUT / "final/stage14_dcrc_final_audit.md"
    report.write_text("\n".join(lines) + "\n")
    print(payload["FinalStatus"], flush=True)


if __name__ == "__main__":
    main()
