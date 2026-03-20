"""
Unit tests for parse_tool_calls() from agent_from_scratch.py

Run: python -m pytest tests/test_parse_tool_calls.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock unsloth before importing agent_from_scratch (not installed locally)
sys.modules["unsloth"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from agent_from_scratch import parse_tool_calls


class TestParseToolCalls:
    def test_single_tool_call(self):
        text = '[TOOL_CALLS] [{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}]'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "get_stock_quote"
        assert result[0]["arguments"] == {"ticker": "AAPL"}

    def test_multiple_tool_calls(self):
        text = '[TOOL_CALLS] [{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}, {"name": "get_financial_ratios", "arguments": {"ticker": "AAPL"}}]'
        result = parse_tool_calls(text)
        assert len(result) == 2
        assert result[0]["name"] == "get_stock_quote"
        assert result[1]["name"] == "get_financial_ratios"

    def test_no_tool_calls(self):
        text = "Apple's P/E ratio is 28.5 based on trailing twelve months."
        result = parse_tool_calls(text)
        assert result == []

    def test_think_block_before_tool_calls(self):
        text = '<think>\nI need to look up AAPL data.\n</think>\n\n[TOOL_CALLS] [{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}]'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "get_stock_quote"

    def test_extra_text_after_json(self):
        text = '[TOOL_CALLS] [{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}] Some extra text the model generated'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "get_stock_quote"

    def test_arguments_as_string(self):
        text = '[TOOL_CALLS] [{"name": "get_stock_quote", "arguments": "{\\"ticker\\": \\"AAPL\\"}"}]'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["arguments"] == {"ticker": "AAPL"}

    def test_empty_string(self):
        result = parse_tool_calls("")
        assert result == []

    def test_malformed_json(self):
        text = "[TOOL_CALLS] [{broken json here"
        result = parse_tool_calls(text)
        assert result == []

    def test_nested_array_in_arguments(self):
        text = '[TOOL_CALLS] [{"name": "get_economic_indicators", "arguments": {"indicators": ["cpi", "unemployment"]}}]'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["arguments"]["indicators"] == ["cpi", "unemployment"]
