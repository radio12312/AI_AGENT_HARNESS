"""Run (case x trial) jobs concurrently and write runs.jsonl + per-run traces."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .agent import Agent
from .checks import evaluate
from .config import resolve
from .failure_tags import tag_failure
from .llm import InfraError, LLMClient
from .stats import reference_cost
from .tools import ToolEnv

JUDGE_PROMPT = """You are grading a customer-support reply. Criteria:
{criteria}

Reply:
\"\"\"{answer}\"\"\"

Answer with exactly one word, PASS or FAIL, on the first line, then one short sentence why."""


def make_judge(llm: LLMClient):
    """LLM judge, used only for cases that explicitly set `judge: true`."""
    def judge(case: dict, answer: str) -> tuple[bool, str]:
        prompt = JUDGE_PROMPT.format(criteria=case.get("judge_criteria", ""), answer=answer)
        try:
            res = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        except InfraError as e:
            return False, f"judge infra error: {e}"
        verdict = res.content.strip()
        return verdict.upper().startswith("PASS"), f"judge: {verdict[:200]}"
    return judge


def run_one(case: dict, trial: int, cfg: dict, llm: LLMClient, out_dir: Path, judge=None) -> dict:
    """One run: fresh fixture -> agent -> checks -> tags -> trace file. Returns the JSONL record."""
    max_steps = int(case["expected"].get("max_steps") or cfg.get("max_steps", 10))
    env = ToolEnv.from_fixture(resolve(cfg["fixture"]), case.get("fault_injection"),
                               float(cfg.get("fault_timeout_s", 0)))   # reset: re-read from disk every run
    agent = Agent(llm, tool_mode=cfg.get("tool_mode", "native"), max_steps=max_steps)
    run = agent.run(case, env, max_steps=max_steps)
    state = env.state()

    infra = run.stop_reason == "infra_error"
    report = evaluate(case, run, state, max_steps, judge=judge if case.get("judge") else None)
    passed = None if infra else report.passed
    tags = tag_failure(case, run, report)
    cost = reference_cost(run.prompt_tokens, run.completion_tokens, cfg.get("pricing", {}))

    run.trace.update({"trial": trial, "category": case["category"], "pass": passed,
                      "failed_checks": [] if infra else report.strings(), "failure_tags": tags,
                      "expected": case["expected"], "fault_injection": case.get("fault_injection")})
    trace_rel = Path("traces") / f"{_safe(case['id'])}__t{trial}.json"
    (out_dir / trace_rel).parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / trace_rel, "w", encoding="utf-8") as f:
        json.dump(run.trace, f, indent=2, ensure_ascii=False, default=str)

    return {
        "case_id": case["id"], "category": case["category"], "trial": trial,
        "pass": passed, "infra_error": infra,
        "failed_checks": [] if infra else report.strings(), "failure_tags": tags,
        "steps": run.steps, "prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens,
        "reasoning_tokens": run.reasoning_tokens, "total_tokens": run.prompt_tokens + run.completion_tokens,
        "cost_usd": round(cost, 8), "latency": round(run.wall_time_s, 3), "llm_latency": round(run.llm_latency_s, 3),
        "retries": run.retries, "stop_reason": run.stop_reason, "error": run.error,
        "trace_path": trace_rel.as_posix(), "final_answer": run.final_answer,
        "model": cfg["model"], "tool_mode": cfg.get("tool_mode", "native"),
    }


def run_eval(cases: list[dict], cfg: dict, llm: LLMClient, out_dir: Path, trials: int,
             concurrency: int, log=print) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, t) for t in range(1, trials + 1) for c in cases]
    judge = make_judge(llm) if any(c.get("judge") for c in cases) else None
    lock = threading.Lock()
    records: list[dict] = []
    started = time.time()
    with open(out_dir / "runs.jsonl", "w", encoding="utf-8") as jsonl, \
            ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(run_one, c, t, cfg, llm, out_dir, judge): (c, t) for c, t in jobs}
        for fut in as_completed(futures):
            case, trial = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 - a bug in the harness itself; record it, keep going
                rec = {"case_id": case["id"], "category": case["category"], "trial": trial, "pass": False,
                       "infra_error": False, "failed_checks": [f"completion: harness crash {e}"],
                       "failure_tags": ["crash"], "steps": 0, "prompt_tokens": 0, "completion_tokens": 0,
                       "reasoning_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "latency": 0.0,
                       "llm_latency": 0.0, "retries": 0, "stop_reason": "crash", "error": repr(e),
                       "trace_path": None, "final_answer": "", "model": cfg["model"],
                       "tool_mode": cfg.get("tool_mode", "native")}
            with lock:
                records.append(rec)
                jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jsonl.flush()
                status = "INFRA" if rec["infra_error"] else ("PASS " if rec["pass"] else "FAIL ")
                extra = "" if rec["pass"] or rec["infra_error"] else f"  tags={','.join(rec['failure_tags'])}"
                log(f"[{len(records):>3}/{len(jobs)}] {status} {rec['case_id']:<28} t{trial} "
                    f"steps={rec['steps']} tok={rec['total_tokens']} retries={rec['retries']} "
                    f"{rec['latency']:.1f}s{extra}")
    log(f"done: {len(records)} runs in {time.time() - started:.0f}s -> {out_dir / 'runs.jsonl'}")
    return records


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
