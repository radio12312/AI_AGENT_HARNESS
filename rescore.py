"""Re-apply the current checks and failure tags to a finished run's traces (no API calls).

Use when a case's expectations or a checker were corrected after a run, so that
before/after runs are scored by the same yardstick:

  python rescore.py --run-dir results/X

The original records are kept in runs.original.jsonl (first rescore only).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from harness.agent import AgentRun
from harness.cases import load_cases
from harness.checks import evaluate
from harness.config import load_config, resolve
from harness.failure_tags import tag_failure


def run_from_trace(trace: dict) -> AgentRun:
    steps = trace.get("steps", [])
    return AgentRun(
        final_answer=trace.get("final_answer") or "",
        stop_reason=trace.get("stop_reason", "crash"),
        steps=len(steps),
        tool_calls=[tr for s in steps for tr in s.get("tool_results", [])],
        events=[ev for s in steps for ev in s.get("events", [])],
        error=trace.get("error"),
        trace=trace,
    )


def rescore(run_dir, cases_glob: str | None = None) -> tuple[int, int]:
    """Returns (records changed, records total)."""
    cfg = load_config()
    cases = {c["id"]: c for c in load_cases(cases_glob or cfg["cases"])}
    path = run_dir / "runs.jsonl"
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not (run_dir / "runs.original.jsonl").exists():
        shutil.copy(path, run_dir / "runs.original.jsonl")

    changed = 0
    for rec in records:
        if rec["infra_error"] or not rec.get("trace_path") or rec["case_id"] not in cases:
            continue
        case = cases[rec["case_id"]]
        trace_file = run_dir / rec["trace_path"]
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
        run = run_from_trace(trace)
        max_steps = int(case["expected"].get("max_steps") or cfg.get("max_steps", 10))
        report = evaluate(case, run, trace.get("final_state", {}), max_steps)
        tags = tag_failure(case, run, report)
        if (report.passed, report.strings(), tags) != (rec["pass"], rec["failed_checks"], rec["failure_tags"]):
            changed += 1
        rec.update(**{"pass": report.passed, "failed_checks": report.strings(), "failure_tags": tags})
        trace.update({"pass": report.passed, "failed_checks": report.strings(), "failure_tags": tags,
                      "expected": case["expected"]})
        trace_file.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return changed, len(records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases", default=None)
    args = ap.parse_args(argv)
    changed, total = rescore(resolve(args.run_dir), args.cases)
    print(f"rescored {total} runs; {changed} changed outcome/checks/tags. Re-run report.py to refresh report.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
