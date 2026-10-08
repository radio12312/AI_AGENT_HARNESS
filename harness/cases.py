"""Load and validate eval cases from cases/*.yaml."""
from __future__ import annotations

import glob as _glob
from pathlib import Path

import yaml

from .config import resolve
from .tools import FAULT_TYPES, TOOL_NAMES

CATEGORIES = ("happy_path", "edge_case", "policy", "adversarial", "tool_failure")
REQUIRED = ("id", "category", "user_message", "expected")


class CaseError(ValueError):
    pass


def load_cases(patterns: str | list[str]) -> list[dict]:
    """Load cases matching one or more globs (a list, or a comma-separated string)."""
    if isinstance(patterns, str):
        patterns = [p.strip() for p in patterns.split(",") if p.strip()]
    paths: set[Path] = set()
    for pattern in patterns:
        matches = _glob.glob(str(resolve(pattern)))
        if not matches:
            raise CaseError(f"no case files match {pattern!r}")
        paths.update(Path(m) for m in matches)
    cases = [load_case(p) for p in sorted(paths)]
    ids = [c["id"] for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise CaseError(f"duplicate case ids: {sorted(dupes)}")
    return sorted(cases, key=lambda c: c["id"])


def load_case(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        case = yaml.safe_load(f)
    validate_case(case, str(path))
    case["_path"] = str(path)
    return case


def validate_case(case: dict, where: str = "<case>") -> None:
    if not isinstance(case, dict):
        raise CaseError(f"{where}: case must be a mapping")
    for key in REQUIRED:
        if key not in case:
            raise CaseError(f"{where}: missing required key {key!r}")
    if case["category"] not in CATEGORIES:
        raise CaseError(f"{where}: category must be one of {CATEGORIES}")
    exp = case["expected"]
    if "expected_tool_calls" not in exp:
        raise CaseError(f"{where}: expected.expected_tool_calls is required (use [] for none)")
    if "expected_state" not in exp:
        raise CaseError(f"{where}: expected.expected_state is required")
    for spec in _flatten_calls(exp.get("expected_tool_calls", []) + exp.get("forbidden_tool_calls", [])):
        if spec.get("name") not in TOOL_NAMES:
            raise CaseError(f"{where}: unknown tool {spec.get('name')!r} in expected/forbidden calls")
    for fault in case.get("fault_injection") or []:
        if fault.get("tool") not in TOOL_NAMES or fault.get("type") not in FAULT_TYPES:
            raise CaseError(f"{where}: bad fault_injection entry {fault}")


def _flatten_calls(specs: list) -> list[dict]:
    out = []
    for s in specs:
        out.extend(s["any_of"] if isinstance(s, dict) and "any_of" in s else [s])
    return out
