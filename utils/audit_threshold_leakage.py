#!/usr/bin/env python3
"""
Audit validation/falsification protocols for leaked hardcoded thresholds.

This script flags assignments and keyword arguments in protocol files where
criterion-carrying literals appear to bypass `utils.falsification_thresholds`.
It is intentionally conservative: findings are candidates for review, not all
of them are guaranteed bugs.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = [ROOT / "Validation", ROOT / "Falsification"]
SUSPICIOUS_NAMES = {
    "alpha",
    "threshold",
    "tau",
    "delta_pi",
    "effect_size_threshold",
    "baseline_threshold",
    "spike_threshold",
    "falsification_threshold",
}
PLOTTING_CALLS = {
    "plot",
    "scatter",
    "bar",
    "barh",
    "hist",
    "grid",
    "fill_between",
    "axhline",
    "axvline",
    "imshow",
    "text",
}


def _is_float_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for directory in TARGET_DIRS:
        files.extend(sorted(directory.rglob("*.py")))
    return files


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _find_leaks(path: Path) -> list[tuple[int, str, float]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings: list[tuple[int, str, float]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_name = target.id.lower()
                    if target_name in SUSPICIOUS_NAMES and _is_float_constant(
                        node.value
                    ):
                        findings.append(
                            (node.lineno, target.id, float(node.value.value))
                        )
        elif isinstance(node, ast.Call):
            call_name = _call_name(node)
            for kw in node.keywords:
                if (
                    kw.arg
                    and kw.arg.lower() in SUSPICIOUS_NAMES
                    and _is_float_constant(kw.value)
                ):
                    if kw.arg.lower() == "alpha" and call_name in PLOTTING_CALLS:
                        continue
                    findings.append((node.lineno, kw.arg, float(kw.value.value)))

    return sorted(findings)


def main() -> int:
    total = 0
    for path in _iter_python_files():
        findings = _find_leaks(path)
        if not findings:
            continue
        print(path.relative_to(ROOT))
        for lineno, name, value in findings:
            print(f"  L{lineno}: {name} = {value}")
            total += 1

    print(f"\nTotal candidate threshold leaks: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
