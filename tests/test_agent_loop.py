"""Agent loop + LLM client behaviour with a scripted fake API (no network)."""
import httpx
import openai
import pytest

from harness.agent import Agent
from harness.llm import InfraError, backoff_delay
from harness.tools import ToolEnv

from conftest import completion, fake_llm, tool_call

CASE = {"id": "t", "category": "happy_path", "auth_email": "brian.okafor@example.com",
        "user_message": "Refund ORD-1003 please", "expected": {"expected_tool_calls": [], "expected_state": {}}}


def _status_error(status: int):
    req = httpx.Request("POST", "https://example.test/v1/chat/completions")
    resp = httpx.Response(status, request=req, headers={"retry-after": "1"})
    cls = {429: openai.RateLimitError, 503: openai.InternalServerError, 401: openai.AuthenticationError,
           400: openai.BadRequestError}[status]
    return cls(f"HTTP {status}", response=resp, body=None)


def test_native_loop_executes_tools_and_finishes(db):
    llm, api = fake_llm([
        completion(tool_calls=[tool_call("get_order", {"order_id": "ORD-1003"})]),
        completion(tool_calls=[tool_call("issue_refund", {"order_id": "ORD-1003", "amount": 89.5, "reason": "broken"})]),
        completion(content="<think>done</think>I refunded $89.50.", reasoning_content=None),
    ])
    env = ToolEnv.from_db(db)
    run = Agent(llm, "native").run(CASE, env)
    assert run.stop_reason == "final_answer" and run.final_answer == "I refunded $89.50."
    assert run.steps == 3 and [c["name"] for c in run.tool_calls] == ["get_order", "issue_refund"]
    assert env.state()["refunds"] == [{"order_id": "ORD-1003", "amount": 89.5}]
    assert run.prompt_tokens == 300 and run.completion_tokens == 60 and run.reasoning_tokens > 0
    assert "tools" in api.calls[0]
    # trace keeps the stripped reasoning; the tool result message reaches the model
    assert run.trace["steps"][2]["assistant"]["reasoning"] == "done"
    assert any(m["role"] == "tool" for m in api.calls[1]["messages"])


def test_json_mode_parses_text_tool_calls(db):
    llm, api = fake_llm([
        completion(content='{"tool": "get_order", "arguments": {"order_id": "ORD-1003"}}'),
        completion(content="Your speaker order was delivered on 2026-09-25."),
    ])
    run = Agent(llm, "json").run(CASE, ToolEnv.from_db(db))
    assert run.stop_reason == "final_answer" and run.tool_calls[0]["ok"]
    assert "tools" not in api.calls[0]
    assert "TOOLS" in api.calls[0]["messages"][0]["content"]
    assert api.calls[1]["messages"][-1]["content"].startswith("TOOL_RESULT get_order")


def test_json_mode_malformed_call_is_reported_back(db):
    llm, _ = fake_llm([
        completion(content='{"tool": "get_order", "arguments": {"order_id": "ORD-1003"'),
        completion(content="Sorry, done."),
    ])
    run = Agent(llm, "json").run(CASE, ToolEnv.from_db(db))
    assert [e["type"] for e in run.events] == ["malformed_tool_call"]
    assert run.steps == 2


def test_native_invalid_arguments_are_malformed(db):
    llm, _ = fake_llm([completion(tool_calls=[tool_call("get_order", '{"order_id": ')]), completion(content="ok")])
    run = Agent(llm, "native").run(CASE, ToolEnv.from_db(db))
    assert run.events[0]["type"] == "malformed_tool_call"
    assert run.tool_calls[0]["error"] == "invalid_arguments"


def test_step_limit(db):
    llm, _ = fake_llm([completion(tool_calls=[tool_call("search_kb", {"query": "x"}, f"c{i}")]) for i in range(3)])
    run = Agent(llm, "native", max_steps=3).run(CASE, ToolEnv.from_db(db))
    assert run.stop_reason == "max_steps" and run.steps == 3


def test_retries_are_logged_but_not_steps(db):
    llm, _ = fake_llm([_status_error(429), _status_error(503), completion(content="hello")])
    run = Agent(llm, "native").run(CASE, ToolEnv.from_db(db))
    assert run.stop_reason == "final_answer" and run.steps == 1 and run.retries == 2
    assert len(run.trace["steps"][0]["retries"]) == 2


def test_retries_exhausted_is_infra_error(db):
    llm, _ = fake_llm([_status_error(503)] * 3, retry={"max_retries": 2, "base_delay_s": 0, "max_delay_s": 0})
    run = Agent(llm, "native").run(CASE, ToolEnv.from_db(db))
    assert run.stop_reason == "infra_error" and run.retries == 2


def test_auth_error_is_infra_without_retry(db):
    llm, api = fake_llm([_status_error(401)])
    with pytest.raises(InfraError):
        llm.chat([{"role": "user", "content": "x"}])
    assert len(api.calls) == 1


def test_bad_request_is_a_crash(db):
    llm, _ = fake_llm([_status_error(400)])
    run = Agent(llm, "native").run(CASE, ToolEnv.from_db(db))
    assert run.stop_reason == "crash" and "BadRequestError" in run.error


def test_backoff_grows_and_respects_cap_and_retry_after():
    assert backoff_delay(0, 2, 60) <= 2
    assert 30 <= backoff_delay(10, 2, 60) <= 60
    assert backoff_delay(0, 2, 60, retry_after=20) == 20
