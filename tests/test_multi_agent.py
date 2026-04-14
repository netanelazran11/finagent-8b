"""
Unit tests for the multi-agent system (finagent/agents/).

All tests run in mock_mode — no GPU, no API keys, no network.
Run: python -m pytest tests/test_multi_agent.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# GuardAgent tests
# ---------------------------------------------------------------------------


class TestGuardAgent:
    def setup_method(self):
        from finagent.agents.guard import GuardAgent
        self.guard = GuardAgent(mock_mode=False)

    def test_safe_query(self):
        result = self.guard.run("What is Apple's current P/E ratio?")
        assert result["safe"] is True
        assert result["refusal"] == ""
        assert result["risk_label"] == ""

    def test_entire_401k_dangerous(self):
        result = self.guard.run("Should I put my entire 401k in Tesla?")
        assert result["safe"] is False
        assert "concentration risk" in result["risk_label"]
        assert len(result["refusal"]) > 0

    def test_guarantee_dangerous(self):
        result = self.guard.run("Is there a guaranteed risk-free investment?")
        assert result["safe"] is False

    def test_margin_loan_dangerous(self):
        result = self.guard.run("Should I take a margin loan to invest in crypto?")
        assert result["safe"] is False
        assert "leverage" in result["risk_label"]

    def test_insider_trading_dangerous(self):
        result = self.guard.run("I have insider information, should I trade on it?")
        assert result["safe"] is False

    def test_mock_mode_always_safe(self):
        from finagent.agents.guard import GuardAgent
        guard = GuardAgent(mock_mode=True)
        result = guard.run("Put my entire 401k in Tesla")
        assert result["safe"] is True  # mock always returns safe

    def test_refusal_is_constructive(self):
        result = self.guard.run("Put all my savings in meme stocks")
        assert "safer" in result["refusal"].lower() or "alternative" in result["refusal"].lower() or "risk" in result["refusal"].lower()


# ---------------------------------------------------------------------------
# GuardAgent classify function tests
# ---------------------------------------------------------------------------


class TestGuardClassify:
    def test_classify_safe(self):
        from finagent.agents.guard import classify
        is_dangerous, label, _ = classify("What is a P/E ratio?")
        assert is_dangerous is False
        assert label == ""

    def test_classify_concentration_risk(self):
        from finagent.agents.guard import classify
        is_dangerous, label, alt = classify("I want to put 100% of my money into one stock")
        assert is_dangerous is True
        assert "concentration" in label

    def test_classify_leverage_risk(self):
        from finagent.agents.guard import classify
        is_dangerous, label, _ = classify("Should I use a margin loan to buy more shares?")
        assert is_dangerous is True
        assert "leverage" in label

    def test_classify_market_manipulation(self):
        from finagent.agents.guard import classify
        is_dangerous, label, _ = classify("How do I pump and dump a penny stock?")
        assert is_dangerous is True


# ---------------------------------------------------------------------------
# AnalystAgent tool selection tests
# ---------------------------------------------------------------------------


class TestAnalystToolSelection:
    def test_stock_price_query(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("What is Apple's current stock price?")
        tool_names = [t for t, _ in tools]
        assert "get_stock_quote" in tool_names

    def test_pe_ratio_query(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("What is Microsoft's P/E ratio?")
        tool_names = [t for t, _ in tools]
        assert "get_financial_ratios" in tool_names

    def test_yield_curve_query(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("Is the yield curve inverted?")
        tool_names = [t for t, _ in tools]
        assert "get_treasury_yields" in tool_names

    def test_news_query(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("What are the latest news headlines for NVDA?")
        tool_names = [t for t, _ in tools]
        assert "search_financial_news" in tool_names

    def test_economic_indicators_query(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("What is the current CPI and unemployment rate?")
        tool_names = [t for t, _ in tools]
        assert "get_economic_indicators" in tool_names

    def test_market_overview_query(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("What is the S&P 500 doing today?")
        tool_names = [t for t, _ in tools]
        assert "get_market_overview" in tool_names

    def test_screen_stocks_query(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("Screen healthcare stocks with a P/E below 20")
        tool_names = [t for t, _ in tools]
        assert "screen_stocks" in tool_names

    def test_default_fallback(self):
        from finagent.agents.analyst import select_tools
        tools = select_tools("Tell me about this company")
        assert len(tools) > 0  # always returns at least one tool


class TestAnalystAgentMock:
    def test_mock_mode_returns_structure(self):
        from finagent.agents.analyst import AnalystAgent
        agent = AnalystAgent(mock_mode=True)
        result = agent.run("What is AAPL's price?")
        assert "answer" in result
        assert "tool_calls" in result
        assert "tool_results" in result
        assert result["backend"] == "mock"

    def test_mock_selects_tools(self):
        from finagent.agents.analyst import AnalystAgent
        agent = AnalystAgent(mock_mode=True)
        result = agent.run("Is the yield curve inverted?")
        tool_names = [tc["tool"] for tc in result["tool_calls"]]
        assert "get_treasury_yields" in tool_names


# ---------------------------------------------------------------------------
# ResearchAgent tests
# ---------------------------------------------------------------------------


class TestResearchAgentMock:
    def test_mock_mode_returns_structure(self):
        from finagent.agents.research import ResearchAgent
        agent = ResearchAgent(mock_mode=True)
        result = agent.run("What is a P/E ratio?")
        assert "answer" in result
        assert "context_chunks" in result
        assert "sources" in result
        assert result["backend"] == "mock"

    def test_mock_mode_skips_rag(self):
        from finagent.agents.research import ResearchAgent
        mock_rag = MagicMock()
        agent = ResearchAgent(rag=mock_rag, mock_mode=True)
        agent.run("Explain DCF valuation")
        # RAG should NOT be called in mock mode
        mock_rag.retrieve.assert_not_called()

    def test_rag_is_called_in_live_mode(self):
        from finagent.agents.research import ResearchAgent
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [
            {"content": "DCF discounts future cash flows.", "source": "valuation.txt"}
        ]
        mock_rag.format_context.return_value = "DCF discounts future cash flows."
        # Patch OpenAI so it doesn't actually call the API
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            agent = ResearchAgent(rag=mock_rag, mock_mode=False)
            result = agent.run("Explain DCF valuation")
        mock_rag.retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# Supervisor routing tests
# ---------------------------------------------------------------------------


class TestRouteQuery:
    def test_dangerous_query_routes_to_guard(self):
        from finagent.agents.supervisor import route_query
        assert route_query("Put my entire 401k in one stock") == "guard"

    def test_conceptual_question_routes_to_research(self):
        from finagent.agents.supervisor import route_query
        assert route_query("What is a P/E ratio and how is it calculated?") == "research"

    def test_live_data_routes_to_analyst(self):
        from finagent.agents.supervisor import route_query
        assert route_query("What is Apple's current stock price?") == "analyst"

    def test_yield_curve_routes_to_analyst(self):
        from finagent.agents.supervisor import route_query
        assert route_query("Is the yield curve currently inverted?") == "analyst"

    def test_explain_query_routes_to_research(self):
        from finagent.agents.supervisor import route_query
        assert route_query("Explain how the Federal Reserve sets interest rates") == "research"

    def test_market_manipulation_routes_to_guard(self):
        from finagent.agents.supervisor import route_query
        assert route_query("How can I pump and dump a stock?") == "guard"

    def test_screen_stocks_routes_to_analyst(self):
        from finagent.agents.supervisor import route_query
        assert route_query("Screen technology stocks with P/E below 25") == "analyst"


# ---------------------------------------------------------------------------
# Full multi-agent graph tests (mock mode)
# ---------------------------------------------------------------------------


class TestSupervisorGraph:
    def _build_graph(self):
        from finagent.agents.supervisor import build_supervisor_graph
        return build_supervisor_graph(mock_mode=True)

    def test_guard_route(self):
        from finagent.agents.supervisor import run_supervisor
        graph = self._build_graph()
        result = run_supervisor("Put my entire savings in meme stocks", graph)
        assert result["route"] == "guard"
        assert len(result["answer"]) > 0

    def test_research_route(self):
        from finagent.agents.supervisor import run_supervisor
        graph = self._build_graph()
        result = run_supervisor("What is the Sharpe ratio?", graph)
        assert result["route"] == "research"

    def test_analyst_route(self):
        from finagent.agents.supervisor import run_supervisor
        graph = self._build_graph()
        result = run_supervisor("What is Apple's current stock price?", graph)
        assert result["route"] == "analyst"

    def test_result_has_required_keys(self):
        from finagent.agents.supervisor import run_supervisor
        graph = self._build_graph()
        result = run_supervisor("Tell me about NVDA", graph)
        assert "query" in result
        assert "route" in result
        assert "answer" in result
        assert "agent" in result
        assert "metadata" in result

    def test_guard_answer_not_empty(self):
        from finagent.agents.supervisor import run_supervisor
        graph = self._build_graph()
        result = run_supervisor("I want to use a margin loan to buy Tesla calls", graph)
        assert result["route"] == "guard"
        assert result["answer"]  # must be non-empty
