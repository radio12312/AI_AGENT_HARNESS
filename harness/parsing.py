"""Parsing model output: reasoning stripping and prompt-based (JSON) tool calls."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>(.*?)</\1>", re.S | re.I)
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning)>", re.I)
_THINK_CLOSE = re.compile(r"</(think|thinking|reasoning)>", re.I)
_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S | re.I)
_FENCE = re.compile(r"```(?:json|tool_call|tool)?\s*(.*?)```", re.S | re.I)
_LOOKS_LIKE_CALL = re.compile(r'"(tool|tool_name|tool_calls|function)"\s*:|<tool_call>', re.I)


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    id: str | None = None
    raw_arguments: str | None = None


class ToolCallParseError(ValueError):
    pass


def split_reasoning(text: str | None) -> tuple[str, str]:
    """Split model text into (answer, reasoning).

    Handles closed <think>...</think> blocks, a dangling </think> (chat templates
    that open the block in the prompt), and an opened-but-never-closed <think>.
    """
    if not text:
        return "", ""
    reasoning: list[str] = [m.group(2).strip() for m in _THINK_BLOCK.finditer(text)]
    answer = _THINK_BLOCK.sub("", text)

    close = _THINK_CLOSE.search(answer)
    if close:
        reasoning.append(answer[: close.start()].strip())
        answer = answer[close.end():]
    opened = _THINK_OPEN.search(answer)
    if opened:
        reasoning.append(answer[opened.end():].strip())
        answer = answer[: opened.start()]
    return answer.strip(), "\n".join(r for r in reasoning if r).strip()


def parse_arguments(raw) -> dict:
    """Parse a tool call's arguments (native mode gives a JSON string)."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        raise ToolCallParseError(f"arguments are not valid JSON: {e}") from e
    if isinstance(value, str):  # double-encoded
        return parse_arguments(value)
    if not isinstance(value, dict):
        raise ToolCallParseError(f"arguments must be a JSON object, got {type(value).__name__}")
    return value


def _iter_json_values(text: str):
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                i += 1
                continue
            yield value
            i = end
        else:
            i += 1


def _normalize(value) -> list[ToolCall]:
    if isinstance(value, list):
        calls: list[ToolCall] = []
        for item in value:
            calls.extend(_normalize(item))
        return calls
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("tool_calls"), list):
        return _normalize(value["tool_calls"])
    if isinstance(value.get("function"), dict):  # OpenAI-shaped {"function": {"name", "arguments"}}
        return _normalize(value["function"])
    name = value.get("tool") or value.get("tool_name")
    if name is None and any(k in value for k in ("arguments", "args", "parameters")):
        name = value.get("name")  # bare "name" only counts next to an arguments key
    if not isinstance(name, str) or not name.strip():
        return []
    args = value.get("arguments", value.get("args", value.get("parameters", {})))
    return [ToolCall(name=name.strip(), arguments=parse_arguments(args))]


def extract_tool_calls(text: str) -> tuple[list[ToolCall], str | None]:
    """Find tool calls written as JSON in plain text (JSON tool mode).

    Returns (calls, error). `error` is set when the text clearly tries to make a
    tool call but it cannot be parsed - a malformed tool call.
    """
    if not text:
        return [], None
    candidates = [m.group(1) for m in _TOOL_CALL_TAG.finditer(text)]
    candidates += [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    last_error = None
    for candidate in candidates:
        for value in _iter_json_values(candidate):
            try:
                calls = _normalize(value)
            except ToolCallParseError as e:
                last_error = str(e)
                continue
            if calls:
                return calls, None
    if _LOOKS_LIKE_CALL.search(text):
        return [], last_error or "text looks like a tool call but no valid JSON tool call was found"
    return [], None
