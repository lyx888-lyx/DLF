#!/usr/bin/env python
"""Create the final Stage23A-v2a artifact/SHA ledger."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from stage23a_v2_common import ROOT, RUNTIME_ROOT, V2_ROOT, atomic_json, git_head, sha256_file


def main():
    analysis = V2_ROOT / "analysis_v2a"
    excluded = {
        analysis / "artifact_manifest.tsv",
        analysis / "artifact_manifest.json",
    }
    roots = [
        V2_ROOT / "features",
        analysis,
        V2_ROOT / "final_v2a",
        V2_ROOT / "protocol" / "v2a_authorization_manifest.json",
        V2_ROOT / "protocol" / "strong_static_protocol.json",
        V2_ROOT / "protocol" / "effective_availability_protocol.json",
        RUNTIME_ROOT / "state.json",
        ROOT / "runtime" / "stage23a_v2a",
    ]
    paths = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    rows = []
    for path in sorted(set(paths)):
        if path in excluded:
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(rows)
    tsv_path = analysis / "artifact_manifest.tsv"
    frame.to_csv(tsv_path, sep="\t", index=False)
    signal = json.loads(
        (analysis / "stage23a_v2a_signal_audit.json").read_text(encoding="utf-8")
    )
    payload = {
        "stage": "Stage23A-v2a final artifact ledger",
        "status": signal["status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "artifact_count_excluding_manifest_files": len(frame),
        "total_bytes": int(frame["bytes"].sum()),
        "artifact_tsv_path": str(tsv_path.resolve()),
        "artifact_tsv_sha256": sha256_file(tsv_path),
        "parent_frozen_manifest_sha256": sha256_file(
            V2_ROOT / "protocol" / "frozen_protocol_manifest.json"
        ),
        "authorization_manifest_sha256": sha256_file(
            V2_ROOT / "protocol" / "v2a_authorization_manifest.json"
        ),
        "formal_judge_trained": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "student_trained": False,
    }
    json_path = analysis / "artifact_manifest.json"
    atomic_json(json_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
