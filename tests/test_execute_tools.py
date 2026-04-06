"""
Unit tests for execute_tools() from agent_from_scratch.py

Run: python -m pytest tests/test_execute_tools.py -v
"""

from unittest.mock import MagicMock, patch

# Mock unsloth before importing agent_from_scratch (not installed locally)
import sys
sys.modules["unsloth"] = MagicMock()
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()

from agent_from_scratch import execute_tools  # noqa: E402


class TestExecuteTools:
    def test_single_tool(self):
        """Known tool returns {"name": ..., "result": ...}"""
        fake_registry = {"get_stock_quote": lambda ticker: {"price": 264.72, "symbol": ticker}}
        with patch("agent_from_scratch.TOOL_REGISTRY", fake_registry):
            result = execute_tools([{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}])
        assert len(result) == 1
        assert result[0]["name"] == "get_stock_quote"
        assert result[0]["result"]["price"] == 264.72

    def test_multiple_tools(self):
        """Multiple tool calls return one result each, in order."""
        fake_registry = {
            "get_stock_quote": lambda ticker: {"price": 264.72},
            "get_financial_ratios": lambda ticker: {"pe_ratio": 33.5},
        }
        with patch("agent_from_scratch.TOOL_REGISTRY", fake_registry):
            result = execute_tools([
                {"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}},
                {"name": "get_financial_ratios", "arguments": {"ticker": "AAPL"}},
            ])
        assert len(result) == 2
        assert result[0]["name"] == "get_stock_quote"
        assert result[1]["name"] == "get_financial_ratios"
        assert result[1]["result"]["pe_ratio"] == 33.5

    def test_unknown_tool(self):
        """Unknown tool name returns {"error": "Unknown tool: ..."}"""
        with patch("agent_from_scratch.TOOL_REGISTRY", {}):
            result = execute_tools([{"name": "fake_tool", "arguments": {}}])
        assert len(result) == 1
        assert "error" in result[0]["result"]
        assert "fake_tool" in result[0]["result"]["error"]

    def test_tool_raises_exception(self):
        """Tool that raises returns {"error": str(e)}"""
        def broken_tool(**kwargs):
            raise ValueError("API is down")

        with patch("agent_from_scratch.TOOL_REGISTRY", {"broken": broken_tool}):
            result = execute_tools([{"name": "broken", "arguments": {}}])
        assert len(result) == 1
        assert "error" in result[0]["result"]
        assert "API is down" in result[0]["result"]["error"]

    def test_empty_list(self):
        """No tool calls → empty results."""
        result = execute_tools([])
        assert result == []

    def test_tool_with_multiple_arguments(self):
        """Tool receives all keyword arguments correctly."""
        def mock_screen(sector, max_pe=None):
            return {"sector": sector, "max_pe": max_pe, "matches": []}

        with patch("agent_from_scratch.TOOL_REGISTRY", {"screen_stocks": mock_screen}):
            result = execute_tools([
                {"name": "screen_stocks", "arguments": {"sector": "Technology", "max_pe": 30}}
            ])
        assert result[0]["result"]["sector"] == "Technology"
        assert result[0]["result"]["max_pe"] == 30
