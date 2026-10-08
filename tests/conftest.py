"""Shared test helpers: a scripted fake LLM so no test ever calls the API."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.config import load_config  # noqa: E402
from harness.llm import LLMClient  # noqa: E402


def tool_call(name: str, args: dict | str, call_id: str = "call_1"):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(name=name, arguments=raw))


def completion(content: str | None = None, tool_calls=None, prompt_tokens=100, completion_tokens=20,
               reasoning_content=None, finish_reason="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=reasoning_content,
                          model_extra={})
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                            completion_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], usage=usage)


class ScriptedCompletions:
    """Returns queued responses in order; an Exception in the queue is raised instead."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))  # snapshot: the agent keeps appending to messages
        if not self.script:
            raise AssertionError("scripted LLM ran out of responses")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_llm(script, **cfg_overrides) -> tuple[LLMClient, ScriptedCompletions]:
    cfg = load_config()
    cfg.update({"rate_limit_rpm": 0, **cfg_overrides})
    completions = ScriptedCompletions(script)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(cfg, api_key="test", client=client, sleep=lambda s: None), completions


@pytest.fixture
def db():
    with open(ROOT / "fixtures" / "db.json", encoding="utf-8") as f:
        return json.load(f)
