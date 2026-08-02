"""Run a V9.25 script with SciPy Spearman-result compatibility.

Older SciPy releases expose ``SpearmanrResult.correlation`` while newer
releases expose ``SignificanceResult.statistic``. V9.25 only needs the
correlation coefficient, so this runner provides both names without changing
SciPy's numerical result.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import scipy.stats as scipy_stats


class _CompatibleSpearmanResult:
    """Tuple-compatible proxy exposing both old and new attribute names."""

    def __init__(self, result) -> None:
        statistic = getattr(result, "statistic", None)
        if statistic is None:
            statistic = getattr(result, "correlation", result[0])
        pvalue = getattr(result, "pvalue", result[1])
        self.statistic = statistic
        self.correlation = statistic
        self.pvalue = pvalue

    def __iter__(self):
        return iter((self.statistic, self.pvalue))

    def __getitem__(self, index):
        return (self.statistic, self.pvalue)[index]

    def __len__(self) -> int:
        return 2


def _install_compatibility() -> None:
    original = scipy_stats.spearmanr

    def compatible_spearmanr(*args, **kwargs):
        result = original(*args, **kwargs)
        if hasattr(result, "statistic"):
            return result
        return _CompatibleSpearmanResult(result)

    scipy_stats.spearmanr = compatible_spearmanr


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python3 scripts/v925_scipy_compat_runner.py "
            "<target-script> [arguments ...]"
        )
    target = Path(sys.argv[1])
    if not target.is_file():
        raise FileNotFoundError(target)
    remaining = sys.argv[2:]
    _install_compatibility()
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
