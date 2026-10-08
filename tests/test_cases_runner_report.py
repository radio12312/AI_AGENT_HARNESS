"""Case files are valid, and runner -> runs.jsonl -> report.md works end to end (fake LLM)."""
import json
from collections import Counter

from harness.cases import load_cases
from harness.config import load_config
from harness.runner import run_eval
from harness.stats import reference_cost, wilson_interval

import report as report_mod
from conftest import completion, fake_llm, tool_call

MIN_PER_CATEGORY = {"happy_path": 8, "edge_case": 6, "policy": 4, "adversarial": 3, "tool_failure": 3}


def test_all_cases_load_and_meet_category_minimums():
    cases = load_cases("cases/*.yaml")
    assert len(cases) >= 24
    counts = Counter(c["category"] for c in cases)
    for cat, n in MIN_PER_CATEGORY.items():
        assert counts[cat] >= n, f"{cat}: {counts[cat]} < {n}"


def test_case_expectations_reference_real_fixture_data(db):
    orders = {o["order_id"] for o in db["orders"]}
    emails = {c["email"] for c in db["customers"]}
    for case in load_cases("cases/*.yaml"):
        assert case.get("auth_email") in emails, case["id"]
        for r in case["expected"]["expected_state"].get("refunds", []):
            assert r["order_id"] in orders, case["id"]


def test_wilson_interval():
    lo, hi = wilson_interval(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_reference_cost():
    assert reference_cost(1_000_000, 1_000_000, {"input_per_1m_usd": 0.5, "output_per_1m_usd": 2.2}) == 2.7


def test_resume_keeps_completed_and_requeues_infra(tmp_path):
    from harness.runner import load_for_resume
    rows = [{"case_id": "a", "trial": 1, "infra_error": False, "pass": True},
            {"case_id": "b", "trial": 1, "infra_error": True, "pass": None}]
    (tmp_path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    kept, infra = load_for_resume(tmp_path)
    assert [r["case_id"] for r in kept] == ["a"] and [r["case_id"] for r in infra] == ["b"]
    assert len((tmp_path / "runs.jsonl").read_text().strip().splitlines()) == 1
    assert "\"b\"" in (tmp_path / "infra_attempts.jsonl").read_text()


def test_runner_and_report_end_to_end(tmp_path):
    cases = [c for c in load_cases("cases/*.yaml") if c["id"] in ("happy_03_simple_refund", "policy_02_duplicate_refund")]
    by_id = {c["id"]: c for c in cases}
    good_refund = [
        completion(tool_calls=[tool_call("get_order", {"order_id": "ORD-1003"})]),
        completion(tool_calls=[tool_call("issue_refund", {"order_id": "ORD-1003", "amount": 89.5, "reason": "broken"})]),
        completion(content="Done - I refunded $89.50 to your card."),
    ]
    bad_duplicate = [
        completion(tool_calls=[tool_call("get_order", {"order_id": "ORD-1006"})]),
        completion(tool_calls=[tool_call("issue_refund", {"order_id": "ORD-1006", "amount": 59, "reason": "again"})]),
        completion(content="Refund of $59.00 issued."),
    ]
    llm, _ = fake_llm(good_refund + bad_duplicate)
    cfg = load_config()
    records = run_eval([by_id["happy_03_simple_refund"], by_id["policy_02_duplicate_refund"]], cfg, llm,
                       tmp_path, trials=1, concurrency=1, log=lambda *_: None)
    recs = {r["case_id"]: r for r in records}
    assert recs["happy_03_simple_refund"]["pass"] is True
    dup = recs["policy_02_duplicate_refund"]
    assert dup["pass"] is False and "policy_violation" in dup["failure_tags"]
    for key in ("case_id", "trial", "pass", "failed_checks", "steps", "prompt_tokens", "completion_tokens",
                "cost_usd", "latency", "retries", "trace_path"):
        assert key in dup
    assert (tmp_path / dup["trace_path"]).exists()
    assert len((tmp_path / "runs.jsonl").read_text().strip().splitlines()) == 2

    (tmp_path / "meta.json").write_text(json.dumps({"model": "m", "tool_mode": "native", "trials": 1,
                                                    "pricing": cfg["pricing"]}))
    md = report_mod.build_report(tmp_path)
    assert "50.0%" in md and "Wilson" in md and "policy_violation" in md
    assert "Reference cost per successful run" in md and "Tokens per successful run" in md
    assert "policy_02_duplicate_refund" in md
