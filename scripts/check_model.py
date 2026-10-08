"""Confirm the model ID, plain chat, and tool calling (native, with JSON fallback) work.

  python scripts/check_model.py [--model MODEL]

Exit code 0 if plain chat works and at least one tool mode works.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent import JSON_MODE_INSTRUCTIONS, tool_list_text  # noqa: E402
from harness.config import api_key, load_config  # noqa: E402
from harness.llm import InfraError, LLMClient  # noqa: E402
from harness.parsing import ToolCallParseError, extract_tool_calls, parse_arguments  # noqa: E402
from harness.tools import TOOL_SCHEMAS  # noqa: E402

QUESTION = "What is the status of order ORD-1001? Look it up with a tool."


def _usage(res) -> str:
    est = " (estimated)" if res.reasoning_tokens_estimated else ""
    return (f"prompt={res.prompt_tokens} completion={res.completion_tokens} "
            f"reasoning={res.reasoning_tokens}{est} latency={res.latency_s:.1f}s retries={len(res.retries)}")


def check_plain(llm: LLMClient) -> bool:
    print("\n[1] plain chat")
    try:
        res = llm.chat([{"role": "user", "content": "Reply with exactly the word: pong"}])
    except InfraError as e:
        print(f"  FAIL (infra): {e}")
        return False
    print(f"  answer: {res.content!r}")
    print(f"  reasoning returned: {'yes, ' + str(len(res.reasoning)) + ' chars (stripped from answer)' if res.reasoning else 'no'}")
    print(f"  {_usage(res)}")
    ok = "pong" in res.content.lower()
    print("  OK" if ok else "  WARN: answer did not contain 'pong'")
    return bool(res.content.strip())


def check_native(llm: LLMClient) -> bool:
    print("\n[2] native tool calling (tools=...)")
    try:
        res = llm.chat([{"role": "system", "content": "You are a support agent. Use tools to look things up."},
                        {"role": "user", "content": QUESTION}], tools=TOOL_SCHEMAS)
    except Exception as e:  # noqa: BLE001 - a 400 here usually means tools are unsupported
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False
    print(f"  finish_reason={res.finish_reason} {_usage(res)}")
    if not res.tool_calls:
        print(f"  FAIL: no tool_calls returned. content={res.content[:200]!r}")
        return False
    tc = res.tool_calls[0]
    try:
        args = parse_arguments(tc["arguments"])
    except ToolCallParseError as e:
        print(f"  FAIL: tool call arguments did not parse: {e}")
        return False
    print(f"  tool call: {tc['name']}({json.dumps(args)})")
    ok = tc["name"] == "get_order" and str(args.get("order_id", "")).upper() == "ORD-1001"
    print("  OK" if ok else "  WARN: unexpected tool/args, but native tool calling works")
    return True


def check_json(llm: LLMClient) -> bool:
    print("\n[3] JSON fallback tool calling (tools described in the system prompt)")
    system = "You are a support agent." + JSON_MODE_INSTRUCTIONS.format(tool_list=tool_list_text())
    try:
        res = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": QUESTION}])
    except InfraError as e:
        print(f"  FAIL (infra): {e}")
        return False
    calls, err = extract_tool_calls(res.content)
    print(f"  {_usage(res)}")
    if not calls:
        print(f"  FAIL: could not parse a tool call ({err}). content={res.content[:200]!r}")
        return False
    print(f"  parsed tool call: {calls[0].name}({json.dumps(calls[0].arguments)})")
    print("  OK")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(overrides={"model": args.model})
    llm = LLMClient(cfg, api_key(cfg))
    print(f"model={cfg['model']} base_url={cfg['base_url']} temperature={cfg.get('temperature')}")

    try:
        ids = {m.id for m in llm.client.models.list().data}
        print(f"[0] model listed by API: {'yes' if cfg['model'] in ids else 'NO (may still work)'}")
    except Exception as e:  # noqa: BLE001 - listing is informational only
        print(f"[0] could not list models: {type(e).__name__}: {e}")

    plain = check_plain(llm)
    native = check_native(llm) if plain else False
    json_ok = check_json(llm) if plain else False

    print("\nSUMMARY")
    print(f"  plain chat:          {'OK' if plain else 'FAIL'}")
    print(f"  native tool calling: {'OK' if native else 'FAIL'}")
    print(f"  JSON fallback:       {'OK' if json_ok else 'FAIL'}")
    if native:
        rec = "native"
    elif json_ok:
        rec = "json"
    else:
        print("  -> no working tool mode")
        return 1
    print(f"  -> recommended tool_mode: {rec} (config.yaml currently: {cfg.get('tool_mode')})")
    return 0 if plain else 1


if __name__ == "__main__":
    sys.exit(main())
