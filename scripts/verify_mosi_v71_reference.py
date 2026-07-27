"""Verify that a local V7.1 MOSI run matches the frozen reference result."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("result/complementarity_v71/mosi/seed_1111")
REFERENCE = Path("repro/mosi_anchored_complementarity_v71.json")
TOLERANCE = 5e-4


def main():
    expected = json.loads(REFERENCE.read_text(encoding="utf-8"))
    actual = json.loads(
        (ROOT / "complementarity_v71_summary.json").read_text(encoding="utf-8")
    )
    target = expected["test_metrics"]["valid_selected_hybrid"]["MAE"]
    observed = actual["results"]["hybrid_valid_selected"]["metrics"]["MAE"]
    difference = abs(float(observed) - float(target))
    print(f"expected MAE: {target:.6f}")
    print(f"observed MAE: {observed:.6f}")
    print(f"absolute difference: {difference:.6f}")
    if difference > TOLERANCE:
        raise SystemExit(
            f"Reference mismatch: difference {difference:.6f} exceeds {TOLERANCE:.6f}"
        )
    print("MOSI V7.1 reference verified.")


if __name__ == "__main__":
    main()
