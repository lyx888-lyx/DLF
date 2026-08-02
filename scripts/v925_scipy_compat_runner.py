"""Run a V9.25 script with SciPy and repository-path compatibility.

Older SciPy releases expose ``SpearmanrResult.correlation`` while newer
releases expose ``SignificanceResult.statistic``. V9.25 only needs the
correlation coefficient, so this runner provides both names without changing
SciPy's numerical result.

The runner also executes targets from the repository root and inserts that
root into ``sys.path``. This keeps imports such as ``trains.singleTask``
working even though the launcher itself lives under ``scripts/``.
"""

from __future__ import annotations

import os
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


def _install_scipy_compatibility() -> None:
    original = scipy_stats.spearmanr

    def compatible_spearmanr(*args, **kwargs):
        result = original(*args, **kwargs)
        if hasattr(result, "statistic"):
            return result
        return _CompatibleSpearmanResult(result)

    scipy_stats.spearmanr = compatible_spearmanr


def _prepare_repository_path(target_argument: str) -> tuple[Path, Path]:
    repository_root = Path(__file__).resolve().parents[1]
    target = Path(target_argument)
    if not target.is_absolute():
        target = repository_root / target
    target = target.resolve()
    if not target.is_file():
        raise FileNotFoundError(target)

    os.chdir(repository_root)
    root_text = str(repository_root)
    target_parent_text = str(target.parent)
    sys.path[:] = [
        value
        for value in sys.path
        if value not in {root_text, target_parent_text}
    ]
    sys.path.insert(0, root_text)
    if target_parent_text != root_text:
        sys.path.insert(1, target_parent_text)
    return repository_root, target


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python3 scripts/v925_scipy_compat_runner.py "
            "<target-script> [arguments ...]"
        )
    remaining = sys.argv[2:]
    _, target = _prepare_repository_path(sys.argv[1])
    _install_scipy_compatibility()
    sys.argv = [str(target), *remaining]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
