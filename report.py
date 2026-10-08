"""Build results/<timestamp>/report.md from runs.jsonl.

  python report.py                         # latest results dir
  python report.py --run-dir results/X     # a specific run
  python report.py --update-readme         # also refresh the results table in README.md
  python report.py --compare results/A results/B   # before/after table (markdown to stdout)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from harness.cases import CATEGORIES
from harness.config import ROOT, load_config, resolve
from harness.stats import wilson_interval

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


# ---------------------------------------------------------------------------
def latest_run_dir(results_dir: Path) -> Path:
    dirs = sorted(p for p in results_dir.iterdir() if (p / "runs.jsonl").exists())
    if not dirs:
        raise SystemExit(f"no runs.jsonl found under {results_dir}")
    return dirs[-1]


def load_runs(run_dir: Path) -> list[dict]:
    with open(run_dir / "runs.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_meta(run_dir: Path) -> dict:
    p = run_dir / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _avg(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def summarize(runs: list[dict]) -> dict:
    valid = [r for r in runs if not r["infra_error"]]
    passes = [r for r in valid if r["pass"]]
    k, n = len(passes), len(valid)
    lo, hi = wilson_interval(k, n)
    total_tokens = sum(r["total_tokens"] for r in runs)
    total_cost = sum(r["cost_usd"] for r in runs)
    return {
        "runs": len(runs), "valid": n, "passes": k, "rate": k / n if n else 0.0, "ci": (lo, hi),
        "infra": len(runs) - n, "avg_steps": _avg(r["steps"] for r in valid),
        "total_tokens": total_tokens, "prompt_tokens": sum(r["prompt_tokens"] for r in runs),
        "completion_tokens": sum(r["completion_tokens"] for r in runs),
        "reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in runs),
        "total_cost": total_cost, "cost_per_success": total_cost / k if k else None,
        "tokens_per_success": total_tokens / k if k else None,
        "retries": sum(r["retries"] for r in runs),
        "avg_latency": _avg(r["latency"] for r in valid),
    }


def tag_counts(runs: list[dict]) -> list[tuple[str, int, list[str]]]:
    counts: Counter = Counter()
    cases: dict[str, set] = defaultdict(set)
    for r in runs:
        if r["infra_error"] or r["pass"]:
            continue
        for t in r["failure_tags"]:
            counts[t] += 1
            cases[t].add(r["case_id"])
    return [(t, c, sorted(cases[t])) for t, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _money(x) -> str:
    return "n/a" if x is None else f"${x:.5f}"


def _num(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


# ---------------------------------------------------------------------------
def headline_table(s: dict) -> str:
    rows = [
        ("Success rate (passes / valid runs)", f"**{_pct(s['rate'])}** ({s['passes']}/{s['valid']})"),
        ("95% Wilson interval", f"{_pct(s['ci'][0])} - {_pct(s['ci'][1])}"),
        ("Infra errors (excluded from rate)", str(s["infra"])),
        ("Avg steps per run", f"{s['avg_steps']:.2f}"),
        ("Total tokens (prompt / completion / reasoning)",
         f"{s['total_tokens']:,} ({s['prompt_tokens']:,} / {s['completion_tokens']:,} / {s['reasoning_tokens']:,})"),
        ("**Tokens per successful run**", _num(s["tokens_per_success"])),
        ("Total reference cost", _money(s["total_cost"])),
        ("**Reference cost per successful run**", _money(s["cost_per_success"])),
        ("API retries (not counted as steps)", str(s["retries"])),
        ("Avg wall time per run", f"{s['avg_latency']:.1f}s"),
    ]
    return "| Metric | Value |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def category_table(runs: list[dict]) -> str:
    lines = ["| Category | Pass rate | Passes / valid | Avg steps | Avg tokens | Avg ref. cost |",
             "|---|---|---|---|---|---|"]
    for cat in CATEGORIES:
        rs = [r for r in runs if r["category"] == cat and not r["infra_error"]]
        if not rs:
            continue
        k = sum(1 for r in rs if r["pass"])
        lines.append(f"| {cat} | {_pct(k / len(rs))} | {k}/{len(rs)} | {_avg(r['steps'] for r in rs):.2f} | "
                     f"{_avg(r['total_tokens'] for r in rs):,.0f} | {_money(_avg(r['cost_usd'] for r in rs))} |")
    return "\n".join(lines)


def case_table(runs: list[dict]) -> str:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_case[r["case_id"]].append(r)
    lines = ["| Case | Category | Pass k/N | Flaky | Avg steps | Avg tokens | Failure tags |",
             "|---|---|---|---|---|---|---|"]
    for cid in sorted(by_case):
        rs = by_case[cid]
        valid = [r for r in rs if not r["infra_error"]]
        k, n = sum(1 for r in valid if r["pass"]), len(valid)
        flaky = "**FLAKY**" if 0 < k < n else ""
        tags = Counter(t for r in valid for t in r["failure_tags"])
        tag_s = ", ".join(f"{t}x{c}" for t, c in tags.most_common())
        infra = f" (+{len(rs) - n} infra)" if len(rs) > n else ""
        lines.append(f"| {cid} | {rs[0]['category']} | {k}/{n}{infra} | {flaky} | "
                     f"{_avg(r['steps'] for r in valid):.1f} | {_avg(r['total_tokens'] for r in valid):,.0f} | {tag_s} |")
    return "\n".join(lines)


def tags_table(runs: list[dict]) -> str:
    rows = tag_counts(runs)
    if not rows:
        return "_No failures._"
    lines = ["| Tag | Failed runs | Cases |", "|---|---|---|"]
    lines += [f"| {t} | {c} | {', '.join(cs)} |" for t, c, cs in rows]
    return "\n".join(lines)


def _clip(s, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def trace_excerpt(run_dir: Path, rec: dict) -> str:
    out = [f"**{rec['case_id']}** (trial {rec['trial']}) - tags: `{', '.join(rec['failure_tags'])}`",
           f"failed checks: {'; '.join(rec['failed_checks'])}", "```text"]
    trace = {}
    if rec.get("trace_path") and (run_dir / rec["trace_path"]).exists():
        trace = json.loads((run_dir / rec["trace_path"]).read_text(encoding="utf-8"))
    user = next((m["content"] for m in trace.get("messages_initial", []) if m["role"] == "user"), "")
    out.append(f"USER: {_clip(user, 220)}")
    for step in trace.get("steps", []):
        for tr in step.get("tool_results", []):
            args = tr.get("arguments") if tr.get("arguments") is not None else tr.get("raw_arguments")
            fault = f" [FAULT:{tr['fault']}]" if tr.get("fault") else ""
            out.append(f"[step {step['step']}] {tr['name']}({_clip(json.dumps(args), 120)}){fault} -> "
                       f"{_clip(tr.get('result'), 160)}")
        for ev in step.get("events", []):
            out.append(f"[step {step['step']}] EVENT {ev['type']}: {_clip(ev['detail'], 140)}")
    out.append(f"FINAL ({rec['stop_reason']}): {_clip(rec.get('final_answer'), 420)}")
    out.append("```")
    return "\n".join(out)


def excerpts_section(run_dir: Path, runs: list[dict]) -> str:
    parts = []
    for tag, count, _ in tag_counts(runs)[:3]:
        failed = [r for r in runs if not r["infra_error"] and not r["pass"] and tag in r["failure_tags"]]
        picked, seen = [], set()
        for r in sorted(failed, key=lambda r: (r["case_id"], r["trial"])):  # prefer distinct cases
            if r["case_id"] not in seen:
                picked.append(r)
                seen.add(r["case_id"])
        picked = (picked + [r for r in failed if r not in picked])[:2]
        parts.append(f"### `{tag}` ({count} failed runs)\n\n" + "\n\n".join(trace_excerpt(run_dir, r) for r in picked))
    return "\n\n".join(parts) if parts else "_No failures to excerpt._"


def build_report(run_dir: Path) -> str:
    runs, meta = load_runs(run_dir), load_meta(run_dir)
    s = summarize(runs)
    pricing = meta.get("pricing") or load_config().get("pricing", {})
    infra_runs = [f"{r['case_id']} t{r['trial']}: {_clip(r.get('error'), 120)}" for r in runs if r["infra_error"]]
    retried = run_dir / "infra_attempts.jsonl"
    if retried.exists():
        attempts = [json.loads(line) for line in retried.read_text(encoding="utf-8").splitlines() if line.strip()]
        infra_runs.append(f"{len(attempts)} earlier attempt(s) ended in infra_error and were re-run on resume "
                          f"(see infra_attempts.jsonl): " + ", ".join(f"{r['case_id']} t{r['trial']}" for r in attempts))
    trials = meta.get("trials", "?")
    sections = [
        f"# Eval report - {run_dir.name}",
        f"Model `{meta.get('model')}` | tool mode `{meta.get('tool_mode')}` | temperature {meta.get('temperature')} | "
        f"{len({r['case_id'] for r in runs})} cases x {trials} trials | label: {meta.get('label') or '-'}",
        "## Headline", headline_table(s),
        f"_Reference cost_ uses ${pricing.get('input_per_1m_usd')}/1M input and ${pricing.get('output_per_1m_usd')}/1M "
        f"output tokens ({pricing.get('source')}, retrieved {pricing.get('retrieved')}). The NVIDIA hosted API itself "
        "is free; raw tokens per successful run are the provider-independent number. Cost per successful run = "
        "total reference cost / passing runs.",
        "## Infra errors",
        "\n".join(f"- {x}" for x in infra_runs) if infra_runs else "None.",
        "## By category", category_table(runs),
        "## By case", case_table(runs),
        "## Failure tags", tags_table(runs),
        "## Example traces (top 3 tags)", excerpts_section(run_dir, runs),
    ]
    return "\n\n".join(sections) + "\n"


def readme_snippet(run_dir: Path) -> str:
    runs, meta = load_runs(run_dir), load_meta(run_dir)
    s = summarize(runs)
    return "\n\n".join([
        f"Latest run: `{run_dir.name}` - `{meta.get('model')}`, {meta.get('tool_mode')} tool mode, "
        f"{len({r['case_id'] for r in runs})} cases x {meta.get('trials')} trials "
        f"(full report: [results/{run_dir.name}/report.md](results/{run_dir.name}/report.md)).",
        headline_table(s), category_table(runs),
    ])


def update_readme(run_dir: Path, readme: Path = ROOT / "README.md") -> None:
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise SystemExit("README.md has no results markers")
    pre, rest = text.split(README_START, 1)
    _, post = rest.split(README_END, 1)
    readme.write_text(f"{pre}{README_START}\n{readme_snippet(run_dir)}\n{README_END}{post}", encoding="utf-8")


def compare(dir_a: Path, dir_b: Path) -> str:
    ra, rb = load_runs(dir_a), load_runs(dir_b)
    sa, sb = summarize(ra), summarize(rb)
    la, lb = load_meta(dir_a).get("label") or dir_a.name, load_meta(dir_b).get("label") or dir_b.name

    def rate(rs, cat=None):
        v = [r for r in rs if not r["infra_error"] and (cat is None or r["category"] == cat)]
        k = sum(1 for r in v if r["pass"])
        return f"{k}/{len(v)} ({_pct(k / len(v)) if v else 'n/a'})"

    lines = [f"| Metric | Before ({la}) | After ({lb}) |", "|---|---|---|",
             f"| Success rate | {rate(ra)} | {rate(rb)} |",
             f"| 95% Wilson CI | {_pct(sa['ci'][0])}-{_pct(sa['ci'][1])} | {_pct(sb['ci'][0])}-{_pct(sb['ci'][1])} |"]
    lines += [f"| {c} | {rate(ra, c)} | {rate(rb, c)} |" for c in CATEGORIES]
    lines += [f"| Avg steps | {sa['avg_steps']:.2f} | {sb['avg_steps']:.2f} |",
              f"| Tokens per successful run | {_num(sa['tokens_per_success'])} | {_num(sb['tokens_per_success'])} |",
              f"| Ref. cost per successful run | {_money(sa['cost_per_success'])} | {_money(sb['cost_per_success'])} |",
              f"| Infra errors | {sa['infra']} | {sb['infra']} |"]
    ta, tb = dict((t, c) for t, c, _ in tag_counts(ra)), dict((t, c) for t, c, _ in tag_counts(rb))
    for t in sorted(set(ta) | set(tb), key=lambda t: -ta.get(t, 0)):
        lines.append(f"| tag `{t}` | {ta.get(t, 0)} | {tb.get(t, 0)} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate report.md for an eval run")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args(argv)

    if args.compare:
        print(compare(resolve(args.compare[0]), resolve(args.compare[1])))
        return 0
    cfg = load_config()
    run_dir = resolve(args.run_dir) if args.run_dir else latest_run_dir(resolve(cfg.get("results_dir", "results")))
    out = run_dir / "report.md"
    out.write_text(build_report(run_dir), encoding="utf-8")
    print(f"wrote {out}")
    if args.update_readme:
        update_readme(run_dir)
        print("updated README.md results table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
