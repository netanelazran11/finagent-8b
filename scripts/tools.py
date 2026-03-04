#!/usr/bin/env python3
"""
Financial tools — real market data via Alpha Vantage API.

7 tools defined in configs/tool_definitions.json, each returning a JSON-serializable dict.
Both agents (from_scratch and langgraph) import TOOL_REGISTRY for dispatch:
    result = TOOL_REGISTRY["get_stock_quote"](ticker="AAPL")
"""

import json
import os
import time
import traceback
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", verbose=True)

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
ALPHAVANTAGE_BASE = "https://www.alphavantage.co/query"

# -- File-based cache (survives between script runs) --
# Keys are "FUNCTION:SYMBOL", e.g. "GLOBAL_QUOTE:AAPL"
_cache_file = Path(__file__).parent.parent / ".av_cache.json"
CACHE_TTL = 3600  # 60 minutes


def _load_cache() -> dict:
    if not _cache_file.exists() or _cache_file.stat().st_size == 0:
        return {}
    try:
        with open(_cache_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict):
    with open(_cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_alpha_vantage(function: str, symbol: str = "", **extra_params) -> dict:
    """Call Alpha Vantage API with file-based cache.
    Cleans numbered key prefixes ("01. symbol" → "symbol").
    Does NOT cache error responses (rate limit, invalid key, etc.).
    """
    cache_key = f"{function}:{symbol}" if symbol else function
    cache = _load_cache()
    now = time.time()

    if cache_key in cache and (now - cache[cache_key]["timestamp"]) < CACHE_TTL:
        print(f"  [cache hit] {cache_key}")
        return cache[cache_key]["data"]

    print(f"  [API call] {function} {symbol}")
    params = {"function": function, "apikey": ALPHAVANTAGE_API_KEY, **extra_params}
    if symbol:
        params["symbol"] = symbol.upper()

    r = requests.get(ALPHAVANTAGE_BASE, params=params)
    data = r.json()

    def clean_keys(obj):
        if isinstance(obj, dict):
            return {
                (k.split(". ", 1)[1] if ". " in k and k[0].isdigit() else k): clean_keys(v)
                for k, v in obj.items()
            }
        return obj

    data = clean_keys(data)

    if "Information" in data or "Note" in data or "Error Message" in data:
        print(f"  [API error] {data}")
        return data

    cache[cache_key] = {"timestamp": now, "data": data}
    _save_cache(cache)

    return data


def _to_number(v):
    """Convert Alpha Vantage string values to float. "264.72" → 264.72, "0.20%" → 0.2"""
    if not isinstance(v, str):
        return v
    v = v.strip().replace("%", "")
    if v == "None" or v == "-":
        return None
    try:
        return float(v)
    except ValueError:
        return v

# =====================================================================
# TOOL 1: get_stock_quote
# =====================================================================
# Get current price, volume, market cap and daily change.
#
# Uses: GLOBAL_QUOTE (price/volume/change) + OVERVIEW (market_cap, 52-week)
#
# Input:  get_stock_quote("AAPL")
# Output: {"ticker": "AAPL", "price": 264.72, "change_pct": 0.2044,
#          "volume": 41827946, "market_cap": 4050000000000,
#          "day_high": 266.53, "day_low": 260.20,
#          "fifty_two_week_high": 280.0, "fifty_two_week_low": 164.08,
#          "currency": "USD"}
# Error:  {"ticker": "AAPL", "error": "..."}
#
# GLOBAL_QUOTE keys: "symbol", "open", "high", "low", "price",
#   "volume", "latest trading day", "previous close", "change", "change percent"
# OVERVIEW keys: "MarketCapitalization", "52WeekHigh", "52WeekLow", "Currency"
# All values are strings → convert with _to_number()
# =====================================================================

def get_stock_quote(ticker: str) -> dict:
    # 1. Call fetch_alpha_vantage("GLOBAL_QUOTE", ticker)
    # 2. Extract fields from "Global Quote" dict (convert strings → float/int)
    # 3. Call fetch_alpha_vantage("OVERVIEW", ticker)
    # 4. Extract market_cap, 52-week high/low, currency
    # 5. Merge both dicts and return
    # 6. Wrap in try/except
    try:
        global_quote_data = fetch_alpha_vantage("GLOBAL_QUOTE", symbol=ticker).get("Global Quote")
        global_quote_cleaned = {k:_to_number(v) for k, v in global_quote_data.items()}
        print()
        overview_data = fetch_alpha_vantage("OVERVIEW", symbol=ticker)
        overview_keys = ["MarketCapitalization","52WeekHigh","52WeekLow","Currency"]
        returned_overview_data = {k:v for k,v in overview_data.items() if k in overview_keys}
        print(overview_data)

        combined_dict = global_quote_cleaned | returned_overview_data
        returned_combined_data = {k:_to_number(v) for k,v in combined_dict.items()}
        return returned_combined_data
    except Exception as e:
        print(f"Exception detected: {e}")
        traceback.print_exc()


# =====================================================================
# TOOL 2: get_financial_ratios
# =====================================================================
# Get fundamental ratios for a company. Uses same OVERVIEW endpoint
# as get_stock_quote (often already cached).
#
# Input:  get_financial_ratios("AAPL", ratios=["pe_ratio", "roe"])
# Output: {"ticker": "AAPL", "company_name": "Apple Inc.",
#          "sector": "Technology", "industry": "Consumer Electronics",
#          "ratios": {"pe_ratio": 28.5, "roe": 0.147}}
# Error:  {"ticker": "AAPL", "error": "..."}
#
# Steps:
#   1. fetch_alpha_vantage("OVERVIEW", ticker)
#   2. If ratios is None → use all keys from RATIO_MAP
#   3. For each ratio: look up the AV key in RATIO_MAP, get value, convert with _to_number()
#   4. Also extract: data["Name"], data["Sector"], data["Industry"]
#   5. Return dict. Wrap in try/except.
# =====================================================================
RATIO_MAP = {
    "pe_ratio": "PERatio",
    "forward_pe": "ForwardPE",
    "pb_ratio": "PriceToBookRatio",
    "ps_ratio": "PriceToSalesRatioTTM",
    "roe": "ReturnOnEquityTTM",
    "roa": "ReturnOnAssetsTTM",
    "profit_margin": "ProfitMargin",
    "dividend_yield": "DividendYield",
    "beta": "Beta",
}


def get_financial_ratios(ticker: str, ratios: list[str] | None = None) -> dict:
    # TODO: implement
    pass


# =====================================================================
# TOOL 3: search_financial_news
# =====================================================================
# Search recent financial news using Alpha Vantage NEWS_SENTIMENT endpoint.
#
# Input:  search_financial_news("AAPL earnings", days_back=7)
# Output: {"query": "AAPL earnings", "days_back": 7, "num_results": 3,
#          "articles": [{"title": "...", "source": "Reuters",
#                        "url": "https://...", "published": "20250120T153000",
#                        "sentiment": "Bullish"}, ...]}
# Error:  {"query": "...", "error": "..."}
#
# Steps:
#   1. Extract potential tickers from query (alpha words, 1-5 chars, uppercase)
#   2. fetch_alpha_vantage("NEWS_SENTIMENT", tickers=candidate)
#      AV returns: {"feed": [{"title": "...", "source": "...", "url": "...",
#                    "time_published": "20250120T153000",
#                    "overall_sentiment_label": "Bullish"}, ...]}
#   3. Filter articles by date (time_published > cutoff)
#   4. Deduplicate by title, limit to 10
#   5. Return dict. Wrap in try/except.
# =====================================================================

def search_financial_news(query: str, days_back: int = 7) -> dict:
    # TODO: implement
    pass


# =====================================================================
# TOOL 4: get_market_overview
# =====================================================================
# Snapshot of major indices + sector ETFs performance.
# Uses GLOBAL_QUOTE on ETF proxies (SPY, QQQ, etc.) and sector ETFs.
#
# Input:  get_market_overview(include_sectors=True)
# Output: {"timestamp": "2025-02-15T14:30:00",
#          "indices": {"S&P 500": {"symbol": "SPY", "price": 545.0, "change_pct": 0.35}, ...},
#          "sectors": {"Technology": {"etf": "XLK", "price": 210.5, "change_pct": 0.8}, ...}}
#
# Steps:
#   1. For each index in INDICES: fetch_alpha_vantage("GLOBAL_QUOTE", symbol)
#      → extract "price", "change percent" from "Global Quote"
#   2. If include_sectors: same for each ETF in SECTOR_ETFS
#   3. Return dict with timestamp, indices, sectors
#   4. Wrap each ticker in its own try/except (don't crash if one fails)
# =====================================================================

# ETF proxies for major indices (AV doesn't support ^GSPC style symbols)
INDICES = {
    "S&P 500": "SPY",
    "NASDAQ": "QQQ",
    "Dow Jones": "DIA",
    "Russell 2000": "IWM",
    "VIX": "VIXY",
}

# Sector SPDR ETFs
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC",
}


def get_market_overview(include_sectors: bool = True) -> dict:
    # TODO: implement
    pass


# =====================================================================
# TOOL 5: get_treasury_yields
# =====================================================================
# US Treasury yield curve data. Uses AV TREASURY_YIELD endpoint.
#
# Input:  get_treasury_yields()
# Output: {"timestamp": "2025-02-15T14:30:00",
#          "yields": {"3month": 5.25, "5year": 4.10, "10year": 4.38, "30year": 4.55},
#          "spread_10y_3m": -0.87,
#          "curve_status": "inverted (recession signal)"}
#
# Steps:
#   1. For each maturity in TREASURY_MATURITIES:
#      fetch_alpha_vantage("TREASURY_YIELD", maturity=m)
#      AV returns: {"name": "...", "data": [{"date": "2025-02-14", "value": "4.38"}, ...]}
#      → take data[0]["value"] (most recent), convert to float
#   2. Compute spread = yields["10year"] - yields["3month"]
#   3. curve_status: "inverted (recession signal)" if spread < 0, else "normal"
#   4. Return dict.
# =====================================================================

TREASURY_MATURITIES = ["3month", "5year", "10year", "30year"]


def get_treasury_yields() -> dict:
    # TODO: implement
    pass


# =====================================================================
# TOOL 6: screen_stocks
# =====================================================================
# Screen stocks in a sector by fundamental criteria.
# Uses OVERVIEW endpoint for each ticker in predefined sector lists.
#
# Input:  screen_stocks("Technology", max_pe=30, min_market_cap_b=100)
# Output: {"sector": "Technology",
#          "filters_applied": {"max_pe": 30, "min_market_cap_b": 100, ...},
#          "num_matches": 3,
#          "matches": [{"ticker": "AAPL", "name": "Apple Inc.", "price": 187.5,
#                       "pe_ratio": 28.5, "market_cap_b": 2900.0}, ...]}
# Error:  {"sector": "...", "error": "Unknown sector. Available: [...]"}
#
# Steps:
#   1. Find sector (case-insensitive) in SECTOR_TICKERS
#   2. For each ticker: fetch_alpha_vantage("OVERVIEW", sym)
#      → extract PERatio, MarketCapitalization, DividendYield from response
#      → apply filters (skip if value is None and filter is set)
#   3. Return dict with matches
#   4. Wrap each ticker in try/except (don't crash if one fails)
# =====================================================================

SECTOR_TICKERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ADBE", "CRM", "AMD", "INTC"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "EL", "GIS"],
    "Industrials": ["CAT", "UNP", "HON", "BA", "RTX", "DE", "LMT", "GE", "MMM", "UPS"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "VMC", "MLM", "DD"],
    "Communication Services": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "TMUS", "VZ", "T", "CHTR", "EA"],
}


def screen_stocks(
    sector: str,
    max_pe: float | None = None,
    min_market_cap_b: float | None = None,
    min_dividend_yield: float | None = None,
    max_debt_to_equity: float | None = None,
) -> dict:
    # TODO: implement
    pass


# =====================================================================
# TOOL 7: get_economic_indicators
# =====================================================================
# Macroeconomic indicators. Hardcoded values (no free FRED API without key).
#
# Input:  get_economic_indicators(["cpi", "fed_funds_rate"])
# Output: {"indicators": {
#              "cpi": {"name": "Consumer Price Index (YoY)", "value": 2.8,
#                      "unit": "%", "date": "2025-01", "source": "Bureau of Labor Statistics"},
#              "fed_funds_rate": {"name": "Federal Funds Rate", "value": 4.50, ...}},
#          "note": "Values are hardcoded snapshots. For live data, use FRED API."}
#
# Steps:
#   1. For each indicator in the list: look up in ECONOMIC_DATA (lowercase)
#   2. If found → add to result. If not → add {"error": "Unknown. Available: [...]"}
#   3. Return dict with indicators and note.
# =====================================================================

ECONOMIC_DATA = {
    "cpi": {
        "name": "Consumer Price Index (YoY)",
        "value": 2.8,
        "unit": "%",
        "date": "2025-01",
        "source": "Bureau of Labor Statistics",
    },
    "unemployment": {
        "name": "Unemployment Rate",
        "value": 4.0,
        "unit": "%",
        "date": "2025-01",
        "source": "Bureau of Labor Statistics",
    },
    "gdp_growth": {
        "name": "Real GDP Growth (QoQ Annualized)",
        "value": 2.3,
        "unit": "%",
        "date": "2024-Q4",
        "source": "Bureau of Economic Analysis",
    },
    "fed_funds_rate": {
        "name": "Federal Funds Rate (Target Upper)",
        "value": 4.50,
        "unit": "%",
        "date": "2025-01",
        "source": "Federal Reserve",
    },
    "consumer_confidence": {
        "name": "Consumer Confidence Index",
        "value": 104.1,
        "unit": "index",
        "date": "2025-01",
        "source": "Conference Board",
    },
    "pmi": {
        "name": "ISM Manufacturing PMI",
        "value": 49.3,
        "unit": "index",
        "date": "2025-01",
        "source": "ISM",
        "note": "Below 50 = contraction",
    },
}


def get_economic_indicators(indicators: list[str]) -> dict:
    # TODO: implement
    pass


# -- Dispatch table: tool name (str) → function --

TOOL_REGISTRY = {
    "get_stock_quote": get_stock_quote,
    "get_financial_ratios": get_financial_ratios,
    "search_financial_news": search_financial_news,
    "get_market_overview": get_market_overview,
    "get_treasury_yields": get_treasury_yields,
    "screen_stocks": screen_stocks,
    "get_economic_indicators": get_economic_indicators,
}


if __name__ == "__main__":
    print(get_stock_quote("AAPL"))
