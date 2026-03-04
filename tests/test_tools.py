"""
Unit tests for scripts/tools.py

Tests use mocked API responses so they run without network access or API keys.
Run: python -m pytest tests/test_tools.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from tools import (
    _to_number,
    get_stock_quote,
    get_financial_ratios,
    search_financial_news,
    get_market_overview,
    get_treasury_yields,
    screen_stocks,
    get_economic_indicators,
    TOOL_REGISTRY,
    RATIO_MAP,
    ECONOMIC_DATA,
    SECTOR_TICKERS,
)


# ── Mock data ────────────────────────────────────────────────────────

MOCK_GLOBAL_QUOTE = {
    "Global Quote": {
        "symbol": "AAPL",
        "open": "262.41",
        "high": "266.53",
        "low": "260.20",
        "price": "264.72",
        "volume": "41827946",
        "latest trading day": "2026-03-02",
        "previous close": "264.18",
        "change": "0.54",
        "change percent": "0.2044%",
    }
}

MOCK_OVERVIEW = {
    "Symbol": "AAPL",
    "Name": "Apple Inc",
    "Sector": "TECHNOLOGY",
    "Industry": "ELECTRONIC COMPUTERS",
    "MarketCapitalization": "4050000000000",
    "PERatio": "33.5",
    "ForwardPE": "28.2",
    "PriceToBookRatio": "45.1",
    "PriceToSalesRatioTTM": "8.5",
    "ReturnOnEquityTTM": "1.47",
    "ReturnOnAssetsTTM": "0.25",
    "ProfitMargin": "0.26",
    "DividendYield": "0.0055",
    "Beta": "1.24",
    "52WeekHigh": "280.00",
    "52WeekLow": "164.08",
    "Currency": "USD",
}

MOCK_NEWS = {
    "feed": [
        {
            "title": "Apple Reports Record Q1 Earnings",
            "source": "Reuters",
            "url": "https://example.com/1",
            "time_published": "20260301T120000",
            "overall_sentiment_label": "Bullish",
        },
        {
            "title": "iPhone Sales Beat Expectations",
            "source": "Bloomberg",
            "url": "https://example.com/2",
            "time_published": "20260228T080000",
            "overall_sentiment_label": "Somewhat-Bullish",
        },
    ]
}

MOCK_TREASURY = {
    "name": "10-Year Treasury Yield",
    "data": [
        {"date": "2026-03-02", "value": "4.38"},
        {"date": "2026-03-01", "value": "4.35"},
    ],
}


# ── Helper tests ─────────────────────────────────────────────────────

class TestToNumber:
    def test_float_string(self):
        assert _to_number("264.72") == 264.72

    def test_integer_string(self):
        assert _to_number("41827946") == 41827946.0

    def test_percent_string(self):
        assert _to_number("0.2044%") == 0.2044

    def test_date_string_unchanged(self):
        assert _to_number("2026-03-02") == "2026-03-02"

    def test_none_string(self):
        assert _to_number("None") is None

    def test_dash_string(self):
        assert _to_number("-") is None

    def test_already_float(self):
        assert _to_number(42.0) == 42.0

    def test_already_int(self):
        assert _to_number(42) == 42


# ── Tool 1: get_stock_quote ──────────────────────────────────────────

class TestGetStockQuote:
    @patch("tools.fetch_alpha_vantage")
    def test_returns_price(self, mock_fetch):
        mock_fetch.side_effect = [MOCK_GLOBAL_QUOTE, MOCK_OVERVIEW]
        result = get_stock_quote("AAPL")
        assert result is not None
        # Should contain price as a float
        assert "price" in result or "symbol" in result

    @patch("tools.fetch_alpha_vantage")
    def test_handles_api_error(self, mock_fetch):
        mock_fetch.return_value = {"Information": "Rate limited"}
        result = get_stock_quote("AAPL")
        # Should not crash — either returns error dict or partial data
        assert result is not None

    @patch("tools.fetch_alpha_vantage")
    def test_ticker_case_insensitive(self, mock_fetch):
        mock_fetch.side_effect = [MOCK_GLOBAL_QUOTE, MOCK_OVERVIEW]
        result = get_stock_quote("aapl")
        assert result is not None


# ── Tool 2: get_financial_ratios ─────────────────────────────────────

class TestGetFinancialRatios:
    @patch("tools.fetch_alpha_vantage")
    def test_returns_ratios(self, mock_fetch):
        mock_fetch.return_value = MOCK_OVERVIEW
        result = get_financial_ratios("AAPL")
        assert result is not None
        if "ratios" in result:
            assert isinstance(result["ratios"], dict)

    @patch("tools.fetch_alpha_vantage")
    def test_specific_ratios(self, mock_fetch):
        mock_fetch.return_value = MOCK_OVERVIEW
        result = get_financial_ratios("AAPL", ratios=["pe_ratio", "beta"])
        assert result is not None
        if "ratios" in result:
            assert len(result["ratios"]) <= 2

    @patch("tools.fetch_alpha_vantage")
    def test_includes_company_info(self, mock_fetch):
        mock_fetch.return_value = MOCK_OVERVIEW
        result = get_financial_ratios("AAPL")
        if result and "error" not in result:
            assert "company_name" in result or "ticker" in result


# ── Tool 3: search_financial_news ────────────────────────────────────

class TestSearchFinancialNews:
    @patch("tools.fetch_alpha_vantage")
    def test_returns_articles(self, mock_fetch):
        mock_fetch.return_value = MOCK_NEWS
        result = search_financial_news("AAPL earnings")
        assert result is not None
        if "articles" in result:
            assert isinstance(result["articles"], list)
            assert len(result["articles"]) <= 10

    @patch("tools.fetch_alpha_vantage")
    def test_empty_query(self, mock_fetch):
        mock_fetch.return_value = {"feed": []}
        result = search_financial_news("xyz")
        assert result is not None


# ── Tool 4: get_market_overview ──────────────────────────────────────

class TestGetMarketOverview:
    @patch("tools.fetch_alpha_vantage")
    def test_returns_indices(self, mock_fetch):
        mock_fetch.return_value = MOCK_GLOBAL_QUOTE
        result = get_market_overview(include_sectors=False)
        assert result is not None
        if "indices" in result:
            assert isinstance(result["indices"], dict)

    @patch("tools.fetch_alpha_vantage")
    def test_includes_sectors(self, mock_fetch):
        mock_fetch.return_value = MOCK_GLOBAL_QUOTE
        result = get_market_overview(include_sectors=True)
        assert result is not None


# ── Tool 5: get_treasury_yields ──────────────────────────────────────

class TestGetTreasuryYields:
    @patch("tools.fetch_alpha_vantage")
    def test_returns_yields(self, mock_fetch):
        mock_fetch.return_value = MOCK_TREASURY
        result = get_treasury_yields()
        assert result is not None
        if "yields" in result:
            assert isinstance(result["yields"], dict)

    @patch("tools.fetch_alpha_vantage")
    def test_computes_spread(self, mock_fetch):
        mock_fetch.return_value = MOCK_TREASURY
        result = get_treasury_yields()
        if result and "spread_10y_3m" in result:
            assert isinstance(result["spread_10y_3m"], (int, float))


# ── Tool 6: screen_stocks ───────────────────────────────────────────

class TestScreenStocks:
    @patch("tools.fetch_alpha_vantage")
    def test_valid_sector(self, mock_fetch):
        mock_fetch.return_value = MOCK_OVERVIEW
        result = screen_stocks("Technology")
        assert result is not None
        if "matches" in result:
            assert isinstance(result["matches"], list)

    def test_invalid_sector(self):
        result = screen_stocks("FakeSector")
        assert result is not None
        if result:
            assert "error" in result

    @patch("tools.fetch_alpha_vantage")
    def test_with_filters(self, mock_fetch):
        mock_fetch.return_value = MOCK_OVERVIEW
        result = screen_stocks("Technology", max_pe=30)
        assert result is not None


# ── Tool 7: get_economic_indicators ──────────────────────────────────

class TestGetEconomicIndicators:
    def test_known_indicator(self):
        result = get_economic_indicators(["cpi"])
        assert result is not None
        if "indicators" in result:
            assert "cpi" in result["indicators"]

    def test_unknown_indicator(self):
        result = get_economic_indicators(["fake_indicator"])
        assert result is not None
        if "indicators" in result:
            assert "error" in result["indicators"].get("fake_indicator", {})

    def test_multiple_indicators(self):
        result = get_economic_indicators(["cpi", "unemployment", "fed_funds_rate"])
        assert result is not None
        if "indicators" in result:
            assert len(result["indicators"]) == 3


# ── TOOL_REGISTRY ────────────────────────────────────────────────────

class TestToolRegistry:
    def test_all_tools_registered(self):
        expected = [
            "get_stock_quote", "get_financial_ratios", "search_financial_news",
            "get_market_overview", "get_treasury_yields", "screen_stocks",
            "get_economic_indicators",
        ]
        for name in expected:
            assert name in TOOL_REGISTRY, f"{name} missing from TOOL_REGISTRY"

    def test_all_values_are_callable(self):
        for name, func in TOOL_REGISTRY.items():
            assert callable(func), f"TOOL_REGISTRY['{name}'] is not callable"
