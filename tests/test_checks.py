from harness.agent import AgentRun
from harness.checks import arg_matches, check_answer, check_completion, check_state, check_tool_calls, evaluate


def call(name, **args):
    return {"name": name, "arguments": args, "ok": True}


class TestAnswer:
    def test_must_contain_case_insensitive_and_any_of(self):
        exp = {"must_contain": ["SHIPPED", {"any_of": ["1-2 business days", "one to two business days"]}]}
        assert check_answer("It **shipped** and arrives in 1–2 business days.", exp) == []  # en dash normalized

    def test_must_contain_missing(self):
        fails = check_answer("It is on the way.", {"must_contain": ["1Z999AA10123456784"]})
        assert len(fails) == 1 and fails[0].check == "must_contain"

    def test_must_not_contain(self):
        fails = check_answer("Her address is 12 Birch Lane.", {"must_not_contain": ["12 birch lane", "555-0101"]})
        assert [f.check for f in fails] == ["must_not_contain"]

    def test_list_shorthand_is_any_of(self):
        assert check_answer("30-day window", {"must_contain": [["30 days", "30-day"]]}) == []


class TestToolCalls:
    def test_numeric_tolerance_and_string_numbers(self):
        assert arg_matches("89.5", 89.50) and arg_matches(89.499, 89.50) and not arg_matches(89.0, 89.50)
        assert arg_matches("$12.99", 12.99)

    def test_string_args_case_and_space_insensitive(self):
        assert arg_matches(" ord-1003 ", "ORD-1003")

    def test_contains_matchers(self):
        assert arg_matches("express shipping time", {"contains": "express"})
        assert arg_matches("warranty coverage", {"contains_any": ["guarantee", "warranty"]})
        assert arg_matches("C010", {"one_of": ["C009", "C010"]})

    def test_expected_calls_satisfied(self):
        calls = [call("get_order", order_id="ORD-1003"), call("issue_refund", order_id="ORD-1003", amount=89.5, reason="x")]
        exp = {"expected_tool_calls": [{"name": "get_order", "args": {"order_id": "ORD-1003"}},
                                       {"name": "issue_refund", "args": {"order_id": "ORD-1003", "amount": 89.50}}]}
        assert check_tool_calls(calls, exp) == []

    def test_expected_call_wrong_args_flags_name_called(self):
        calls = [call("issue_refund", order_id="ORD-1003", amount=50, reason="x")]
        fails = check_tool_calls(calls, {"expected_tool_calls": [{"name": "issue_refund", "args": {"amount": 89.5}}]})
        assert fails[0].check == "expected_tool_calls" and fails[0].data["name_called"] is True

    def test_expected_any_of(self):
        exp = {"expected_tool_calls": [{"any_of": [{"name": "get_order", "args": {"order_id": "ORD-1001"}},
                                                    {"name": "list_orders", "args": {"customer_id": "C001"}}]}]}
        assert check_tool_calls([call("list_orders", customer_id="C001")], exp) == []
        assert check_tool_calls([call("get_order", order_id="ORD-1O01")], exp)

    def test_forbidden_by_name_and_by_args(self):
        exp = {"forbidden_tool_calls": [{"name": "issue_refund"},
                                        {"name": "list_orders", "args": {"customer_id": "C001"}}]}
        assert check_tool_calls([call("list_orders", customer_id="C012")], exp) == []
        fails = check_tool_calls([call("list_orders", customer_id="c001"), call("issue_refund", order_id="X", amount=1)], exp)
        assert [f.check for f in fails] == ["forbidden_tool_calls", "forbidden_tool_calls"]

    def test_faulted_calls_still_count_as_made(self):
        c = call("get_order", order_id="ORD-1013")
        c.update(ok=False, fault="timeout")
        assert check_tool_calls([c], {"expected_tool_calls": [{"name": "get_order", "args": {"order_id": "ORD-1013"}}]}) == []


class TestState:
    def test_refunds_exact_multiset(self):
        exp = {"refunds": [{"order_id": "ORD-1003", "amount": 89.5}]}
        assert check_state({"refunds": [{"order_id": "ORD-1003", "amount": 89.50}]}, exp) == []
        assert check_state({"refunds": []}, exp)
        dup = {"refunds": [{"order_id": "ORD-1003", "amount": 89.5}] * 2}
        assert check_state(dup, exp)  # refunding twice is a state mismatch

    def test_escalated(self):
        assert check_state({"refunds": [], "escalated": False}, {"escalated": True})[0].data["field"] == "escalated"
        assert check_state({"refunds": [], "escalated": True}, {"refunds": []}) == []  # unchecked when absent


class TestCompletion:
    def test_reasons(self):
        ok = AgentRun(final_answer="hi", stop_reason="final_answer", steps=3)
        assert check_completion(ok, 5) == []
        assert check_completion(AgentRun(stop_reason="max_steps", steps=5), 5)[0].data["reason"] == "max_steps"
        assert check_completion(AgentRun(stop_reason="crash", error="boom"), 5)[0].data["reason"] == "crash"
        assert check_completion(AgentRun(final_answer="  ", stop_reason="final_answer", steps=1), 5)[0].data["reason"] == "empty_answer"


def test_evaluate_full_pass_and_fail():
    case = {"id": "c", "category": "happy_path", "user_message": "u", "expected": {
        "must_contain": ["89.50"], "must_not_contain": [], "forbidden_tool_calls": [{"name": "escalate_to_human"}],
        "expected_tool_calls": [{"name": "issue_refund", "args": {"order_id": "ORD-1003", "amount": 89.5}}],
        "expected_state": {"refunds": [{"order_id": "ORD-1003", "amount": 89.5}], "escalated": False}}}
    run = AgentRun(final_answer="Refunded $89.50.", stop_reason="final_answer", steps=2,
                   tool_calls=[call("issue_refund", order_id="ORD-1003", amount=89.5, reason="broken")])
    state = {"refunds": [{"order_id": "ORD-1003", "amount": 89.5}], "escalated": False}
    assert evaluate(case, run, state, 6).passed

    run.final_answer = "Refund done."
    report = evaluate(case, run, {"refunds": [], "escalated": True}, 6)
    assert not report.passed
    assert {f.check for f in report.failures} == {"must_contain", "expected_state"}


def test_judge_only_when_case_requests_it():
    case = {"id": "c", "category": "happy_path", "user_message": "u", "judge": True, "judge_criteria": "polite",
            "expected": {"expected_tool_calls": [], "expected_state": {}}}
    run = AgentRun(final_answer="ok", stop_reason="final_answer", steps=1)
    assert not evaluate(case, run, {}, 5, judge=lambda c, a: (False, "rude")).passed
    assert evaluate(case, run, {}, 5, judge=lambda c, a: (True, "fine")).passed
    case["judge"] = False
    assert evaluate(case, run, {}, 5, judge=lambda c, a: (False, "never called")).passed
