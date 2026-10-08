"""OpenAI-SDK client for the NVIDIA API with retry/backoff, pacing and reasoning stripping."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

import openai

from .parsing import split_reasoning

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
INFRA_STATUS = {401, 403, 404}


class InfraError(RuntimeError):
    """The API could not serve the request after retries (or auth/model is wrong).

    Runs that end this way are tagged infra_error and excluded from the success rate.
    """

    def __init__(self, message: str, retries: list | None = None):
        super().__init__(message)
        self.retries = retries or []


@dataclass
class ChatResult:
    content: str                                   # answer text, reasoning stripped
    reasoning: str                                 # stripped reasoning (field and/or <think> text)
    tool_calls: list[dict]                         # [{"id", "name", "arguments"(raw str)}]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_tokens_estimated: bool = False
    latency_s: float = 0.0
    retries: list[dict] = field(default_factory=list)
    finish_reason: str | None = None
    raw_content: str | None = None


class RateLimiter:
    """Spaces request starts evenly across all worker threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            time.sleep(start - now)


def backoff_delay(attempt: int, base: float, cap: float, retry_after: float | None = None) -> float:
    """Exponential backoff with equal jitter; honours Retry-After when the server sends one."""
    ceiling = min(cap, base * (2 ** attempt))
    delay = ceiling / 2 + random.uniform(0, ceiling / 2)
    if retry_after:
        delay = max(delay, min(retry_after, cap))
    return delay


class LLMClient:
    def __init__(self, cfg: dict, api_key: str, client=None, sleep=time.sleep):
        self.cfg = cfg
        self.model = cfg["model"]
        retry = cfg.get("retry", {})
        self.max_retries = int(retry.get("max_retries", 6))
        self.base_delay = float(retry.get("base_delay_s", 2.0))
        self.max_delay = float(retry.get("max_delay_s", 60.0))
        self.limiter = RateLimiter(cfg.get("rate_limit_rpm", 0))
        self._sleep = sleep
        # SDK-level retries are disabled: we retry ourselves so every retry is logged.
        self.client = client or openai.OpenAI(
            base_url=cfg["base_url"], api_key=api_key, max_retries=0,
            timeout=float(cfg.get("request_timeout_s", 180)),
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **overrides) -> ChatResult:
        params = {
            "model": overrides.get("model", self.model),
            "messages": messages,
            "temperature": overrides.get("temperature", self.cfg.get("temperature", 0.6)),
            "top_p": self.cfg.get("top_p", 0.95),
            "max_tokens": overrides.get("max_tokens", self.cfg.get("max_tokens", 4096)),
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if self.cfg.get("extra_body"):
            params["extra_body"] = self.cfg["extra_body"]

        retries: list[dict] = []
        last_error = "unknown"
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            t0 = time.perf_counter()
            retry_after = None
            try:
                resp = self.client.chat.completions.create(**params)
                if not getattr(resp, "choices", None):
                    raise _EmptyResponse("response had no choices")
                return _to_result(resp, time.perf_counter() - t0, retries)
            except openai.APIStatusError as e:
                status = e.status_code
                if status in INFRA_STATUS:
                    raise InfraError(f"HTTP {status}: {_short(e)}", retries) from e
                if status not in RETRYABLE_STATUS:
                    raise  # a 4xx caused by our request: a crash, not infra
                last_error = f"HTTP {status}: {_short(e)}"
                retry_after = _retry_after(e)
            except (openai.APIConnectionError, openai.APITimeoutError, _EmptyResponse) as e:
                last_error = f"{type(e).__name__}: {_short(e)}"
            if attempt == self.max_retries:
                break
            delay = backoff_delay(attempt, self.base_delay, self.max_delay, retry_after)
            retries.append({"attempt": attempt + 1, "error": last_error, "sleep_s": round(delay, 2)})
            self._sleep(delay)
        raise InfraError(f"gave up after {self.max_retries} retries: {last_error}", retries)


class _EmptyResponse(RuntimeError):
    pass


def _to_result(resp, latency: float, retries: list) -> ChatResult:
    choice = resp.choices[0]
    msg = choice.message
    extra = getattr(msg, "model_extra", None) or {}
    field_reasoning = (getattr(msg, "reasoning_content", None) or extra.get("reasoning_content")
                       or extra.get("reasoning") or "")
    answer, think_text = split_reasoning(msg.content or "")
    reasoning = "\n".join(x for x in (str(field_reasoning).strip(), think_text) if x)

    tool_calls = []
    for i, tc in enumerate(getattr(msg, "tool_calls", None) or []):
        fn = tc.function
        tool_calls.append({"id": tc.id or f"call_{i}", "name": fn.name, "arguments": fn.arguments})

    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
    estimated = False
    if not reasoning_tokens and reasoning:
        reasoning_tokens = max(1, len(reasoning) // 4)  # ~4 chars/token when the API doesn't report it
        estimated = True

    return ChatResult(
        content=answer, reasoning=reasoning, tool_calls=tool_calls,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens, reasoning_tokens_estimated=estimated,
        latency_s=latency, retries=retries, finish_reason=choice.finish_reason, raw_content=msg.content,
    )


def _retry_after(e: openai.APIStatusError) -> float | None:
    try:
        value = e.response.headers.get("retry-after")
        return float(value) if value else None
    except (AttributeError, ValueError):
        return None


def _short(e: Exception) -> str:
    return str(e).replace("\n", " ")[:300]
