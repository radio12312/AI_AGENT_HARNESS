"""Run the eval set: python run_eval.py --trials 5 --cases "cases/*.yaml" --concurrency 2"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from harness.cases import load_cases
from harness.config import api_key, load_config, resolve
from harness.llm import LLMClient
from harness.runner import load_for_resume, run_eval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Agent Reliability Harness runner")
    ap.add_argument("--trials", type=int, default=5, help="runs per case (default 5)")
    ap.add_argument("--cases", default=None, help="glob(s), comma-separated (default from config)")
    ap.add_argument("--concurrency", type=int, default=None, help="parallel runs (default from config: 2)")
    ap.add_argument("--model", default=None, help="model id override")
    ap.add_argument("--tool-mode", choices=["native", "json"], default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="default step limit override")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--out", default=None, help="results directory (default results/<timestamp>)")
    ap.add_argument("--label", default="", help="free-text label stored in meta.json (e.g. 'baseline')")
    ap.add_argument("--fix", action="append", default=None, metavar="NAME",
                    help="enable an opt-in agent fix (repeatable), e.g. --fix tool_error_policy")
    ap.add_argument("--resume", default=None, metavar="DIR",
                    help="continue an interrupted run: keep its completed runs, re-run missing and infra_error ones")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"model": args.model, "tool_mode": args.tool_mode,
                                    "concurrency": args.concurrency, "max_steps": args.max_steps,
                                    "cases": args.cases, "fixes": args.fix})
    cases = load_cases(cfg["cases"])
    if args.resume:
        out_dir = resolve(args.resume)
    elif args.out:
        out_dir = Path(args.out)
    else:
        out_dir = resolve(cfg.get("results_dir", "results")) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    done: set = set()
    prior_meta = {}
    if args.resume:
        kept, infra = load_for_resume(out_dir)
        ids = {c["id"] for c in cases}
        done = {(r["case_id"], r["trial"]) for r in kept if r["case_id"] in ids}
        if (out_dir / "meta.json").exists():
            prior_meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        print(f"resuming {out_dir}: keeping {len(done)} completed runs, re-running {len(infra)} infra_error runs + missing")

    meta = {
        "label": args.label, "model": cfg["model"], "tool_mode": cfg.get("tool_mode"),
        "fixes": list(cfg.get("fixes") or []),
        "temperature": cfg.get("temperature"), "max_steps_default": cfg.get("max_steps"),
        "trials": args.trials, "concurrency": cfg.get("concurrency"), "cases": [c["id"] for c in cases],
        "pricing": cfg.get("pricing"), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(), "python": platform.python_version(),
    }
    if prior_meta:
        meta["label"] = meta["label"] or prior_meta.get("label", "")
        meta["started"] = prior_meta.get("started", meta["started"])
        meta["resumed"] = prior_meta.get("resumed", []) + [time.strftime("%Y-%m-%d %H:%M:%S")]
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"model={cfg['model']} tool_mode={cfg.get('tool_mode')} cases={len(cases)} trials={args.trials} "
          f"concurrency={cfg.get('concurrency')} -> {out_dir}")
    llm = LLMClient(cfg, api_key(cfg))
    run_eval(cases, cfg, llm, out_dir, args.trials, int(cfg.get("concurrency", 2)), done=done)

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with open(out_dir / "runs.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    valid = [r for r in records if not r["infra_error"]]
    passes = sum(1 for r in valid if r["pass"])
    print(f"pass {passes}/{len(valid)} valid runs; infra_error {len(records) - len(valid)}. "
          f"Run `python report.py --run-dir {out_dir}` for the full report.")
    return 0


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    sys.exit(main())
