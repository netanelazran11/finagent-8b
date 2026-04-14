"""
AnalystAgent — answers data-driven market questions using live Alpha Vantage tools.

Flow:
    query
      → select_tools(query)               # keyword-based tool selection
      → execute each tool in TOOL_REGISTRY
      → synthesize answer from tool results

This reuses the EXACT same TOOL_REGISTRY and execute_tools logic as the single-agent
scripts (agent_from_scratch.py, agent_langgraph.py). The multi-agent architecture
adds routing on top — the tools themselves are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from finagent.tools import (
    TOOL_REGISTRY,
    get_economic_indicators,
    get_financial_ratios,
    get_market_overview,
    get_stock_quote,
    get_treasury_yields,
    screen_stocks,
    search_financial_news,
)

log = logging.getLogger("finagent.agents.analyst")

# Keywords → tool name. Longer / more specific patterns should come first.
_TOOL_KEYWORDS: list[tuple[str, str]] = [
    (r"\b(p/?e|price.to.earn|valuation|roe|roa|ratio|fundamental)", "get_financial_ratios"),
    (r"\b(news|headline|article|sentiment|recent|latest)\b", "search_financial_news"),
    (r"\b(yield curve|treasury|t.bill|10.year|30.year|3.month)\b", "get_treasury_yields"),
    (r"\b(index|indices|s&p|nasdaq|dow|russell|vix|market (overview|snapshot))\b", "get_market_overview"),
    (r"\b(screen|filter|find stocks|sector stocks|which stocks)\b", "screen_stocks"),
    (r"\b(cpi|inflation|unemployment|fed rate|gdp|macro|economic)\b", "get_economic_indicators"),
    (r"\b(price|quote|stock|share|trading|close|open|volume|market cap)\b", "get_stock_quote"),
]

_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

SYNTHESIS_PROMPT = """You are FinAgent, a precise financial analyst.
Below are live market data results retrieved for the user's question.
Synthesize a clear, data-driven answer. Reference the actual numbers.
Highlight any risks or caveats relevant to the data.

TOOL RESULTS:
{results}

USER QUESTION:
{query}

Answer:"""


def _extract_tickers(query: str) -> list[str]:
    """Extract likely ticker symbols from a query (uppercase 1–5 char words)."""
    stopwords = {"I", "A", "THE", "IN", "AT", "ON", "TO", "MY", "BY", "US",
                 "IF", "OR", "BE", "DO", "IS", "IT", "OF", "AND", "FOR", "ARE",
                 "ETF", "API", "CEO", "CFO", "IPO", "AI", "ML"}
    tokens = _TICKER_RE.findall(query.upper())
    return [t for t in tokens if t not in stopwords]


def select_tools(query: str) -> list[tuple[str, dict]]:
    """Keyword-based tool selection.

    Returns a list of (tool_name, kwargs) tuples to execute.
    Extracts tickers and routes to the right combination of tools.
    """
    selected: list[tuple[str, dict]] = []
    q = query.lower()
    tickers = _extract_tickers(query)
    primary_ticker = tickers[0] if tickers else "SPY"

    tool_names: list[str] = []
    for pattern, tool_name in _TOOL_KEYWORDS:
        if re.search(pattern, q, re.IGNORECASE):
            if tool_name not in tool_names:
                tool_names.append(tool_name)

    # Default to get_stock_quote if no tool matched
    if not tool_names:
        tool_names = ["get_stock_quote"]

    # Build kwargs for each selected tool
    for name in tool_names:
        if name == "get_stock_quote":
            selected.append((name, {"ticker": primary_ticker}))
        elif name == "get_financial_ratios":
            selected.append((name, {"ticker": primary_ticker}))
        elif name == "search_financial_news":
            selected.append((name, {"query": query[:120], "days_back": 7}))
        elif name == "get_treasury_yields":
            selected.append((name, {}))
        elif name == "get_market_overview":
            selected.append((name, {"include_sectors": True}))
        elif name == "screen_stocks":
            # Try to extract sector from query
            sectors = [
                "Technology", "Healthcare", "Financials", "Energy",
                "Consumer Discretionary", "Consumer Staples", "Industrials",
                "Utilities", "Real Estate", "Materials", "Communication Services",
            ]
            sector = next((s for s in sectors if s.lower() in q), "Technology")
            selected.append((name, {"sector": sector}))
        elif name == "get_economic_indicators":
            # Select relevant indicators from query keywords
            all_indicators = ["cpi", "unemployment", "gdp_growth", "fed_funds_rate",
                              "consumer_confidence", "pmi"]
            requested = [i for i in all_indicators if i.replace("_", " ") in q or i in q]
            if not requested:
                requested = ["cpi", "fed_funds_rate", "unemployment"]
            selected.append((name, {"indicators": requested}))

    return selected


def execute_selected(selected: list[tuple[str, dict]]) -> list[dict]:
    """Run selected tools and collect results."""
    results = []
    for name, kwargs in selected:
        if name not in TOOL_REGISTRY:
            results.append({"tool": name, "error": f"Unknown tool: {name}"})
            continue
        try:
            result = TOOL_REGISTRY[name](**kwargs)
            results.append({"tool": name, "kwargs": kwargs, "result": result})
            log.info("[analyst] %s(%s) → ok", name, kwargs)
        except Exception as e:
            results.append({"tool": name, "kwargs": kwargs, "error": str(e)})
            log.warning("[analyst] %s failed: %s", name, e)
    return results


class AnalystAgent:
    """Data-driven market analyst powered by Alpha Vantage tools.

    Args:
        model: Loaded fine-tuned model (or None for GPT-4o-mini / mock).
        tokenizer: Paired tokenizer.
        mock_mode: If True, skip all API/LLM calls.
    """

    def __init__(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        mock_mode: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.mock_mode = mock_mode
        self._openai_key = os.getenv("OPENAI_API_KEY")

    def run(self, query: str) -> dict:
        """Run tool selection → execution → synthesis.

        Returns:
            {
                "answer": str,
                "tool_calls": list[dict],   # tools that were called
                "tool_results": list[dict], # raw results
                "backend": str,
            }
        """
        if self.mock_mode:
            return self._mock_response(query)

        selected = select_tools(query)
        tool_results = execute_selected(selected)
        answer = self._synthesize(query, tool_results)

        return {
            "answer": answer,
            "tool_calls": [{"tool": n, "kwargs": k} for n, k in selected],
            "tool_results": tool_results,
            "backend": self._backend_name(),
        }

    def _synthesize(self, query: str, tool_results: list[dict]) -> str:
        results_str = json.dumps(tool_results, indent=2)[:3000]
        if self._openai_key:
            return self._synthesize_openai(query, results_str)
        if self.model is not None:
            return self._synthesize_finagent(query, results_str)
        return f"Retrieved data:\n{results_str}"

    def _synthesize_openai(self, query: str, results_str: str) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": SYNTHESIS_PROMPT.format(
                        results=results_str, query=query
                    )},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning("OpenAI synthesis failed: %s", e)
            return f"Tool results:\n{results_str}"

    def _synthesize_finagent(self, query: str, results_str: str) -> str:
        messages = [
            {"role": "user", "content": SYNTHESIS_PROMPT.format(
                results=results_str, query=query
            )},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.2)
        return self.tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        ).strip()

    def _backend_name(self) -> str:
        if self._openai_key:
            return "gpt-4o-mini"
        if self.model is not None:
            return "finagent"
        return "mock"

    def _mock_response(self, query: str) -> dict:
        selected = select_tools(query)
        return {
            "answer": f"[Mock] Would call: {[n for n, _ in selected]}",
            "tool_calls": [{"tool": n, "kwargs": k} for n, k in selected],
            "tool_results": [],
            "backend": "mock",
        }
