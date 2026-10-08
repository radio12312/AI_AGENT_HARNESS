import pytest

from harness.parsing import ToolCallParseError, extract_tool_calls, parse_arguments, split_reasoning


class TestSplitReasoning:
    def test_closed_think_block(self):
        answer, reasoning = split_reasoning("<think>check the order first</think>Your order shipped.")
        assert answer == "Your order shipped."
        assert reasoning == "check the order first"

    def test_dangling_close_tag(self):
        # Templates that open <think> in the prompt only emit the closing tag.
        answer, reasoning = split_reasoning("I should look this up.\n</think>\n\nIt was delivered.")
        assert answer == "It was delivered."
        assert "look this up" in reasoning

    def test_unclosed_open_tag_is_all_reasoning(self):
        answer, reasoning = split_reasoning("Sure.<think>still thinking and ran out of tokens")
        assert answer == "Sure."
        assert reasoning.startswith("still thinking")

    def test_no_reasoning(self):
        assert split_reasoning("plain answer") == ("plain answer", "")
        assert split_reasoning(None) == ("", "")

    def test_multiple_blocks_and_case_insensitive(self):
        answer, reasoning = split_reasoning("<THINK>a</THINK>x <think>b</think>y")
        assert answer == "x y"
        assert reasoning == "a\nb"


class TestParseArguments:
    def test_json_string(self):
        assert parse_arguments('{"order_id": "ORD-1001"}') == {"order_id": "ORD-1001"}

    def test_empty(self):
        assert parse_arguments("") == {} and parse_arguments(None) == {}

    def test_double_encoded(self):
        assert parse_arguments('"{\\"a\\": 1}"') == {"a": 1}

    @pytest.mark.parametrize("raw", ['{"order_id": "ORD-1001"', "not json", "[1, 2]"])
    def test_invalid(self, raw):
        with pytest.raises(ToolCallParseError):
            parse_arguments(raw)


class TestExtractToolCalls:
    def test_bare_json(self):
        calls, err = extract_tool_calls('{"tool": "get_order", "arguments": {"order_id": "ORD-1001"}}')
        assert err is None
        assert calls[0].name == "get_order" and calls[0].arguments == {"order_id": "ORD-1001"}

    def test_fenced_with_prose(self):
        text = 'Let me check.\n```json\n{"tool": "search_kb", "arguments": {"query": "refund"}}\n```'
        calls, _ = extract_tool_calls(text)
        assert [c.name for c in calls] == ["search_kb"]

    def test_tool_call_tag_and_name_key(self):
        text = '<tool_call>{"name": "list_orders", "arguments": {"customer_id": "C001"}}</tool_call>'
        calls, _ = extract_tool_calls(text)
        assert calls[0].name == "list_orders" and calls[0].arguments == {"customer_id": "C001"}

    def test_arguments_as_string(self):
        calls, _ = extract_tool_calls('{"tool": "get_order", "arguments": "{\\"order_id\\": \\"ORD-1\\"}"}')
        assert calls[0].arguments == {"order_id": "ORD-1"}

    def test_tool_calls_list(self):
        text = '{"tool_calls": [{"tool": "get_order", "arguments": {"order_id": "A"}}, {"tool": "search_kb", "args": {"query": "q"}}]}'
        calls, _ = extract_tool_calls(text)
        assert [c.name for c in calls] == ["get_order", "search_kb"]

    def test_openai_function_shape(self):
        calls, _ = extract_tool_calls('{"function": {"name": "get_order", "arguments": "{\\"order_id\\": \\"X\\"}"}}')
        assert calls[0].name == "get_order" and calls[0].arguments == {"order_id": "X"}

    def test_plain_answer_is_not_a_call(self):
        assert extract_tool_calls("Your order ORD-1001 was delivered on 2026-09-20.") == ([], None)

    def test_json_without_tool_key_is_not_a_call(self):
        calls, err = extract_tool_calls('Here is the data: {"name": "Alice", "tier": "gold"}')
        assert calls == [] and err is None

    def test_malformed_attempt_reports_error(self):
        calls, err = extract_tool_calls('{"tool": "get_order", "arguments": {"order_id": "ORD-1001"')
        assert calls == [] and err

    def test_bad_arguments_string_reports_error(self):
        calls, err = extract_tool_calls('{"tool": "get_order", "arguments": "{oops"}')
        assert calls == [] and "not valid JSON" in err
