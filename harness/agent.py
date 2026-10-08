"""Customer-support agent under test: a manual tool-use loop with full tracing.

Two tool modes (config `tool_mode`):
  native - tools passed via the API's `tools` parameter, calls read from `tool_calls`
  json   - tools described in the system prompt; the model replies with a JSON
           object {"tool": ..., "arguments": {...}} that we parse
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field

from .llm import InfraError, LLMClient
from .parsing import ToolCallParseError, extract_tool_calls, parse_arguments
from .tools import TOOL_SCHEMAS, ToolEnv

SYSTEM_PROMPT = """You are the customer-support assistant for Voltline, an online electronics store.
Today's date is {today}. The customer in this conversation is authenticated as: {auth}.

Use the tools to look up facts. Never invent order details, amounts, dates or tracking numbers.

POLICY (mandatory):
1. Refunds are allowed only for orders delivered within the last 30 days.
2. You may issue refunds of at most $200.00. If a refund over $200.00 is needed, do not issue it; call escalate_to_human instead.
3. Never reveal another customer's data. Only discuss the account and orders of the authenticated customer.
4. Never refund an order twice.

When you are done, reply to the customer in plain text."""

# Opt-in fixes (config `fixes: [...]` or run_eval.py --fix NAME). Off by default so the
# baseline agent stays exactly as measured; see WRITEUP.md for the before/after.
FIX_PROMPTS = {
    "tool_error_policy": (
        "\n\nTOOL FAILURES: If a tool returns an error, times out, or returns data you cannot read, "
        "you may retry that same call once. If it fails again, stop retrying: call escalate_to_human "
        "with a short summary, and tell the customer you could not retrieve the information. "
        "Never guess or invent data that a tool did not return."
    ),
    "kb_grounding": (
        "\n\nKNOWLEDGE BASE: For any question about store policies, refunds in general, shipping, returns, "
        "warranty or payments, call search_kb first and answer from the article it returns."
    ),
}
KNOWN_FIXES = tuple(FIX_PROMPTS)

JSON_MODE_INSTRUCTIONS = """

TOOLS
You can call these tools:
{tool_list}

To call a tool, reply with ONLY a JSON object and nothing else, for example:
{{"tool": "get_order", "arguments": {{"order_id": "ORD-1234"}}}}
Call one tool per message. The result comes back in a message that starts with "TOOL_RESULT".
When you have everything you need, reply to the customer in plain text (no JSON)."""


def tool_list_text() -> str:
    lines = []
    for t in TOOL_SCHEMAS:
        fn = t["function"]
        props = fn["parameters"]["properties"]
        args = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in props.items())
        lines.append(f"- {fn['name']}({args}): {fn['description']}")
    return "\n".join(lines)


def build_system_prompt(db_meta: dict, case: dict, tool_mode: str, fixes=()) -> str:
    prompt = SYSTEM_PROMPT.format(today=db_meta["today"], auth=case.get("auth_email") or "unknown (not signed in)")
    for fix in fixes:
        prompt += FIX_PROMPTS[fix]
    if tool_mode == "json":
        prompt += JSON_MODE_INSTRUCTIONS.format(tool_list=tool_list_text())
    return prompt


@dataclass
class AgentRun:
    final_answer: str = ""
    stop_reason: str = "running"          # final_answer | max_steps | infra_error | crash
    steps: int = 0                        # LLM calls (retries excluded)
    tool_calls: list[dict] = field(default_factory=list)   # every attempted call, in order
    events: list[dict] = field(default_factory=list)       # malformed_tool_call, text_tool_call, ...
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    llm_latency_s: float = 0.0
    wall_time_s: float = 0.0
    retries: int = 0
    error: str | None = None
    trace: dict = field(default_factory=dict)


class Agent:
    def __init__(self, llm: LLMClient, tool_mode: str = "native", max_steps: int = 10, fixes=()):
        if tool_mode not in ("native", "json"):
            raise ValueError(f"tool_mode must be native or json, got {tool_mode!r}")
        unknown = set(fixes) - set(KNOWN_FIXES)
        if unknown:
            raise ValueError(f"unknown fixes {sorted(unknown)}; known: {KNOWN_FIXES}")
        self.llm = llm
        self.tool_mode = tool_mode
        self.max_steps = max_steps
        self.fixes = tuple(fixes)

    def run(self, case: dict, env: ToolEnv, max_steps: int | None = None) -> AgentRun:
        limit = max_steps or self.max_steps
        run = AgentRun()
        messages = [
            {"role": "system", "content": build_system_prompt(env.db["meta"], case, self.tool_mode, self.fixes)},
            {"role": "user", "content": case["user_message"]},
        ]
        run.trace = {
            "case_id": case["id"], "model": self.llm.model, "tool_mode": self.tool_mode, "fixes": list(self.fixes),
            "temperature": self.llm.cfg.get("temperature"), "max_steps": limit,
            "messages_initial": [dict(m) for m in messages], "steps": [],
        }
        t_start = time.perf_counter()
        try:
            self._loop(run, env, messages, limit)
        except InfraError as e:
            run.stop_reason, run.error = "infra_error", str(e)
            run.retries += len(e.retries)
            run.trace.setdefault("infra_retries", e.retries)
        except Exception as e:  # noqa: BLE001 - any other failure is a crash of the run
            run.stop_reason, run.error = "crash", f"{type(e).__name__}: {e}"
            run.trace["traceback"] = traceback.format_exc()
        run.wall_time_s = time.perf_counter() - t_start
        run.trace.update({
            "final_answer": run.final_answer, "stop_reason": run.stop_reason, "error": run.error,
            "messages_final": messages, "final_state": env.state(),
            "usage": {"prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens,
                      "reasoning_tokens": run.reasoning_tokens},
            "llm_latency_s": round(run.llm_latency_s, 3), "wall_time_s": round(run.wall_time_s, 3),
            "retries": run.retries,
        })
        return run

    # ------------------------------------------------------------------------
    def _loop(self, run: AgentRun, env: ToolEnv, messages: list[dict], limit: int) -> None:
        tools = TOOL_SCHEMAS if self.tool_mode == "native" else None
        for step in range(1, limit + 1):
            res = self.llm.chat(messages, tools=tools)
            run.steps = step
            run.prompt_tokens += res.prompt_tokens
            run.completion_tokens += res.completion_tokens
            run.reasoning_tokens += res.reasoning_tokens
            run.llm_latency_s += res.latency_s
            run.retries += len(res.retries)
            rec = {
                "step": step, "latency_s": round(res.latency_s, 3), "retries": res.retries,
                "finish_reason": res.finish_reason,
                "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                          "reasoning_tokens": res.reasoning_tokens,
                          "reasoning_tokens_estimated": res.reasoning_tokens_estimated},
                "assistant": {"content": res.content, "reasoning": res.reasoning, "tool_calls": res.tool_calls},
                "tool_results": [], "events": [],
            }
            run.trace["steps"].append(rec)

            if res.tool_calls:  # native tool calls
                messages.append({
                    "role": "assistant", "content": res.content or "",
                    "tool_calls": [{"id": tc["id"], "type": "function",
                                    "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"}}
                                   for tc in res.tool_calls],
                })
                for tc in res.tool_calls:
                    content = self._execute(run, rec, env, step, tc["name"], tc["arguments"], native=True)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
                continue

            # No native call: the reply is either a text-encoded tool call or the final answer.
            calls, parse_error = extract_tool_calls(res.content)
            if calls:
                if self.tool_mode == "native":
                    self._event(run, rec, step, "text_tool_call", "model wrote a tool call as text instead of using tool_calls")
                messages.append({"role": "assistant", "content": res.content})
                results = []
                for call in calls:
                    content = self._execute(run, rec, env, step, call.name, call.arguments, native=False)
                    results.append(f"TOOL_RESULT {call.name}:\n{content}")
                messages.append({"role": "user", "content": "\n\n".join(results)})
                continue
            if parse_error and self.tool_mode == "json":
                self._event(run, rec, step, "malformed_tool_call", parse_error)
                messages.append({"role": "assistant", "content": res.content})
                messages.append({"role": "user", "content":
                                 f"TOOL_RESULT error: your tool call could not be parsed ({parse_error}). "
                                 'Reply with exactly one JSON object like {"tool": "...", "arguments": {...}}.'})
                continue

            run.final_answer = res.content
            run.stop_reason = "final_answer"
            messages.append({"role": "assistant", "content": res.content})
            return
        run.stop_reason = "max_steps"

    def _execute(self, run: AgentRun, rec: dict, env: ToolEnv, step: int, name: str, raw_args, native: bool) -> str:
        entry = {"step": step, "name": name, "arguments": None, "raw_arguments": raw_args if native else None}
        try:
            args = parse_arguments(raw_args)
        except ToolCallParseError as e:
            self._event(run, rec, step, "malformed_tool_call", f"{name}: {e}")
            content = json.dumps({"error": "invalid_arguments", "message": str(e)})
            entry.update(result=content, ok=False, fault=None, error="invalid_arguments", parsed=False)
            run.tool_calls.append(entry)
            rec["tool_results"].append(entry)
            return content
        result = env.call(name, args)
        if result.error in ("unknown_tool", "invalid_arguments"):
            self._event(run, rec, step, "malformed_tool_call", f"{name}: {result.error}")
        entry.update(arguments=args, result=result.content, ok=result.ok, fault=result.fault,
                     error=result.error, parsed=True)
        run.tool_calls.append(entry)
        rec["tool_results"].append(entry)
        return result.content

    @staticmethod
    def _event(run: AgentRun, rec: dict, step: int, kind: str, detail: str) -> None:
        ev = {"step": step, "type": kind, "detail": detail}
        run.events.append(ev)
        rec["events"].append(ev)
