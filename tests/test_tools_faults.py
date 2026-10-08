import json

import pytest

from harness.tools import Fault, ToolEnv


def env_with(db, faults=None):
    return ToolEnv.from_db(db, faults)


class TestTools:
    def test_get_order_and_not_found(self, db):
        env = env_with(db)
        r = env.call("get_order", {"order_id": "ord-1003 "})
        assert r.ok and json.loads(r.content)["total"] == 89.5
        r = env.call("get_order", {"order_id": "ORD-9999"})
        assert not r.ok and r.error == "not_found"

    def test_deterministic(self, db):
        a, b = env_with(db), env_with(db)
        for name, args in [("search_kb", {"query": "how long do refunds take"}), ("list_orders", {"customer_id": "C015"})]:
            assert a.call(name, args).content == b.call(name, args).content

    def test_search_kb_ranks_relevant_article(self, db):
        res = json.loads(env_with(db).call("search_kb", {"query": "express shipping"}).content)
        assert res["results"][0]["id"] == "KB-002"
        res = json.loads(env_with(db).call("search_kb", {"query": "warranty"}).content)
        assert res["results"][0]["id"] == "KB-004"

    def test_zero_orders(self, db):
        res = json.loads(env_with(db).call("list_orders", {"customer_id": "C013"}).content)
        assert res["orders"] == [] and res["count"] == 0

    def test_refund_records_state_and_is_permissive_on_policy(self, db):
        env = env_with(db)
        assert env.call("issue_refund", {"order_id": "ORD-1006", "amount": 59, "reason": "dup"}).ok  # duplicate allowed
        assert env.state()["refunds"] == [{"order_id": "ORD-1006", "amount": 59.0}]

    def test_refund_rejects_impossible_amounts(self, db):
        env = env_with(db)
        assert env.call("issue_refund", {"order_id": "ORD-1015", "amount": 500, "reason": "x"}).error == "invalid_amount"
        assert env.call("issue_refund", {"order_id": "ORD-1015", "amount": "$0", "reason": "x"}).error == "invalid_amount"
        assert env.state()["refunds"] == []

    def test_escalation_state(self, db):
        env = env_with(db)
        env.call("escalate_to_human", {"summary": "help"})
        assert env.state()["escalated"] is True

    def test_unknown_tool_and_bad_args(self, db):
        env = env_with(db)
        assert env.call("delete_everything", {}).error == "unknown_tool"
        assert env.call("get_order", {"id": "ORD-1001"}).error == "invalid_arguments"

    def test_env_does_not_mutate_source_fixture(self, db):
        env = env_with(db)
        env.call("issue_refund", {"order_id": "ORD-1003", "amount": 10, "reason": "x"})
        assert db["orders"][2]["refunds"] == []
        assert env_with(db).state()["refunds"] == []  # a fresh env is a reset


class TestFaultInjection:
    def test_error_on_first_call_only(self, db):
        env = env_with(db, [{"tool": "get_order", "type": "error", "on_call": 1}])
        first = env.call("get_order", {"order_id": "ORD-1001"})
        assert first.fault == "error" and not first.ok and "503" in first.content
        second = env.call("get_order", {"order_id": "ORD-1001"})
        assert second.ok and second.fault is None

    def test_timeout_on_all_calls(self, db):
        env = env_with(db, [{"tool": "get_order", "type": "timeout", "on_call": "all"}])
        for _ in range(3):
            r = env.call("get_order", {"order_id": "ORD-1013"})
            assert r.fault == "timeout" and "timed out" in r.content
        assert env.call("list_orders", {"customer_id": "C008"}).ok  # other tools unaffected

    def test_malformed_is_invalid_json_and_leaks_nothing(self, db):
        env = env_with(db, [{"tool": "list_orders", "type": "malformed", "on_call": "all"}])
        r = env.call("list_orders", {"customer_id": "C011"})
        assert r.fault == "malformed"
        with pytest.raises(json.JSONDecodeError):
            json.loads(r.content)
        assert "ORD-1020" not in r.content and "ORD-1021" not in r.content

    def test_fault_on_specific_call_numbers(self, db):
        env = env_with(db, [{"tool": "search_kb", "type": "error", "on_call": [2, 3]}])
        results = [env.call("search_kb", {"query": "refund"}).fault for _ in range(4)]
        assert results == [None, "error", "error", None]

    def test_faulted_refund_has_no_side_effect(self, db):
        env = env_with(db, [{"tool": "issue_refund", "type": "timeout", "on_call": "all"}])
        env.call("issue_refund", {"order_id": "ORD-1003", "amount": 89.5, "reason": "x"})
        assert env.state()["refunds"] == []

    @pytest.mark.parametrize("spec", [{"tool": "get_order", "type": "explode"}, {"tool": "nope", "type": "error"}])
    def test_invalid_fault_spec(self, spec):
        with pytest.raises(ValueError):
            Fault.parse(spec)
