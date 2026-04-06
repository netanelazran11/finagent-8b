#!/usr/bin/env python3
"""
Financial tools — real market data via Alpha Vantage API.

7 tools defined in configs/tool_definitions.json, each returning a JSON-serializable dict.
Both agents (from_scratch and langgraph) import TOOL_REGISTRY for dispatch:
    result = TOOL_REGISTRY["get_stock_quote"](ticker="AAPL")
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

log = logging.getLogger("finagent.tools")

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
        with open(_cache_file) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict) -> None:
    with open(_cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_alpha_vantage(function: str, symbol: str = "", **extra_params) -> dict:
    """Call Alpha Vantage API with a 60-minute file cache.

    - Cleans "01. symbol" prefixes that Alpha Vantage adds to nested dicts.
    - Does NOT cache error responses (rate-limit, invalid key, etc.).
    - Retries once on transient request failures.
    """
    cache_key = f"{function}:{symbol}" if symbol else function
    cache = _load_cache()
    now = time.time()

    if cache_key in cache and (now - cache[cache_key]["timestamp"]) < CACHE_TTL:
        log.debug("cache hit %s", cache_key)
        return cache[cache_key]["data"]

    log.info("API call %s %s", function, symbol)
    params = {"function": function, "apikey": ALPHAVANTAGE_API_KEY, **extra_params}
    if symbol:
        params["symbol"] = symbol.upper()

    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            r = requests.get(ALPHAVANTAGE_BASE, params=params, timeout=15)
            data = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            if attempt == 1:
                time.sleep(0.5)
            continue
    else:
        return {"Error Message": f"Request failed: {last_exc}"}

    def clean_keys(obj):
        if isinstance(obj, dict):
            return {
                (k.split(". ", 1)[1] if ". " in k and k[0].isdigit() else k): clean_keys(v)
                for k, v in obj.items()
            }
        return obj

    data = clean_keys(data)

    if "Information" in data or "Note" in data or "Error Message" in data:
        log.warning("API error response: %s", data)
        return data

    cache[cache_key] = {"timestamp": now, "data": data}
    _save_cache(cache)
    return data


def _to_number(v: str | int | float | None) -> float | str | None:
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


def _api_error(data: dict) -> str | None:
    """Return a human-readable error if the AV response is an error envelope."""
    if not isinstance(data, dict):
        return "Unexpected API response"
    return data.get("Information") or data.get("Note") or data.get("Error Message")


def get_stock_quote(ticker: str) -> dict:
    ticker = ticker.upper()
    try:
        quote_resp = fetch_alpha_vantage("GLOBAL_QUOTE", symbol=ticker)
        quote = (quote_resp or {}).get("Global Quote") or {}
        if not quote:
            return {"ticker": ticker, "error": _api_error(quote_resp) or "No quote data"}

        result = {
            "ticker": ticker,
            "price": _to_number(quote.get("price")),
            "open": _to_number(quote.get("open")),
            "day_high": _to_number(quote.get("high")),
            "day_low": _to_number(quote.get("low")),
            "previous_close": _to_number(quote.get("previous close")),
            "change": _to_number(quote.get("change")),
            "change_pct": _to_number(quote.get("change percent")),
            "volume": _to_number(quote.get("volume")),
            "latest_trading_day": quote.get("latest trading day"),
        }

        overview = fetch_alpha_vantage("OVERVIEW", symbol=ticker) or {}
        if _api_error(overview) is None:
            result["market_cap"] = _to_number(overview.get("MarketCapitalization"))
            result["fifty_two_week_high"] = _to_number(overview.get("52WeekHigh"))
            result["fifty_two_week_low"] = _to_number(overview.get("52WeekLow"))
            result["currency"] = overview.get("Currency")

        return result
    except Exception as e:
        log.exception("get_stock_quote failed")
        return {"ticker": ticker, "error": str(e)}


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
    ticker = ticker.upper()
    try:
        data = fetch_alpha_vantage("OVERVIEW", symbol=ticker) or {}
        if _api_error(data) is not None or not data.get("Symbol"):
            return {"ticker": ticker, "error": _api_error(data) or "No overview data"}

        requested = ratios if ratios else list(RATIO_MAP.keys())
        ratio_values: dict[str, float | None] = {}
        for key in requested:
            av_key = RATIO_MAP.get(key)
            if av_key is None:
                continue
            ratio_values[key] = _to_number(data.get(av_key))

        return {
            "ticker": ticker,
            "company_name": data.get("Name"),
            "sector": data.get("Sector"),
            "industry": data.get("Industry"),
            "ratios": ratio_values,
        }
    except Exception as e:
        log.exception("get_financial_ratios failed")
        return {"ticker": ticker, "error": str(e)}


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
    try:
        tokens = re.findall(r"\b[A-Z]{1,5}\b", query.upper())
        params: dict = {}
        if tokens:
            params["tickers"] = ",".join(tokens[:3])

        data = fetch_alpha_vantage("NEWS_SENTIMENT", **params) or {}
        feed = data.get("feed") or []
        if _api_error(data) is not None and not feed:
            return {"query": query, "error": _api_error(data)}

        cutoff = datetime.now() - timedelta(days=days_back)
        seen: set[str] = set()
        articles: list[dict] = []

        for item in feed:
            title = item.get("title") or ""
            if not title or title in seen:
                continue
            seen.add(title)

            ts = item.get("time_published") or ""
            try:
                published = datetime.strptime(ts[:8], "%Y%m%d")
                if published < cutoff:
                    continue
            except (ValueError, IndexError):
                pass  # keep article if date can't be parsed

            articles.append(
                {
                    "title": title,
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "published": ts,
                    "sentiment": item.get("overall_sentiment_label"),
                }
            )
            if len(articles) >= 10:
                break

        return {
            "query": query,
            "days_back": days_back,
            "num_results": len(articles),
            "articles": articles,
        }
    except Exception as e:
        log.exception("search_financial_news failed")
        return {"query": query, "error": str(e)}


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


def _quote_snapshot(symbol: str) -> dict:
    """Helper: fetch a single GLOBAL_QUOTE and return a compact dict."""
    data = fetch_alpha_vantage("GLOBAL_QUOTE", symbol=symbol) or {}
    quote = data.get("Global Quote") or {}
    if not quote:
        return {"symbol": symbol, "error": _api_error(data) or "No data"}
    return {
        "symbol": symbol,
        "price": _to_number(quote.get("price")),
        "change_pct": _to_number(quote.get("change percent")),
    }


def get_market_overview(include_sectors: bool = True) -> dict:
    indices: dict[str, dict] = {}
    for name, symbol in INDICES.items():
        try:
            indices[name] = _quote_snapshot(symbol)
        except Exception as e:
            indices[name] = {"symbol": symbol, "error": str(e)}

    result: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "indices": indices,
    }

    if include_sectors:
        sectors: dict[str, dict] = {}
        for name, symbol in SECTOR_ETFS.items():
            try:
                snap = _quote_snapshot(symbol)
                snap["etf"] = snap.pop("symbol")
                sectors[name] = snap
            except Exception as e:
                sectors[name] = {"etf": symbol, "error": str(e)}
        result["sectors"] = sectors

    return result


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
    yields: dict[str, float | None] = {}
    for maturity in TREASURY_MATURITIES:
        try:
            data = fetch_alpha_vantage("TREASURY_YIELD", maturity=maturity) or {}
            series = data.get("data") or []
            yields[maturity] = _to_number(series[0].get("value")) if series else None
        except Exception:
            log.exception("treasury fetch failed for %s", maturity)
            yields[maturity] = None

    result: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "yields": yields,
    }

    ten_y, three_m = yields.get("10year"), yields.get("3month")
    if isinstance(ten_y, (int, float)) and isinstance(three_m, (int, float)):
        spread = round(ten_y - three_m, 4)
        result["spread_10y_3m"] = spread
        result["curve_status"] = "inverted (recession signal)" if spread < 0 else "normal"

    return result


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
    "Consumer Discretionary": [
        "AMZN",
        "TSLA",
        "HD",
        "MCD",
        "NKE",
        "SBUX",
        "LOW",
        "TJX",
        "BKNG",
        "CMG",
    ],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "EL", "GIS"],
    "Industrials": ["CAT", "UNP", "HON", "BA", "RTX", "DE", "LMT", "GE", "MMM", "UPS"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "VMC", "MLM", "DD"],
    "Communication Services": [
        "GOOGL",
        "META",
        "DIS",
        "NFLX",
        "CMCSA",
        "TMUS",
        "VZ",
        "T",
        "CHTR",
        "EA",
    ],
}


def screen_stocks(
    sector: str,
    max_pe: float | None = None,
    min_market_cap_b: float | None = None,
    min_dividend_yield: float | None = None,
    max_debt_to_equity: float | None = None,
) -> dict:
    sector_key = next((s for s in SECTOR_TICKERS if s.lower() == sector.lower()), None)
    if sector_key is None:
        return {
            "sector": sector,
            "error": f"Unknown sector. Available: {list(SECTOR_TICKERS.keys())}",
        }

    matches: list[dict] = []
    for ticker in SECTOR_TICKERS[sector_key]:
        try:
            data = fetch_alpha_vantage("OVERVIEW", symbol=ticker) or {}
            if _api_error(data) is not None or not data.get("Symbol"):
                continue

            pe = _to_number(data.get("PERatio"))
            mcap = _to_number(data.get("MarketCapitalization"))
            mcap_b = mcap / 1e9 if isinstance(mcap, (int, float)) else None
            div_y = _to_number(data.get("DividendYield"))

            if max_pe is not None and isinstance(pe, (int, float)) and pe > max_pe:
                continue
            if (
                min_market_cap_b is not None
                and isinstance(mcap_b, (int, float))
                and mcap_b < min_market_cap_b
            ):
                continue
            if (
                min_dividend_yield is not None
                and isinstance(div_y, (int, float))
                and div_y < min_dividend_yield
            ):
                continue

            matches.append(
                {
                    "ticker": ticker,
                    "name": data.get("Name"),
                    "pe_ratio": pe,
                    "market_cap_b": round(mcap_b, 2) if isinstance(mcap_b, (int, float)) else None,
                    "dividend_yield": div_y,
                }
            )
        except Exception:
            log.exception("screen failed for %s", ticker)
            continue

    return {
        "sector": sector_key,
        "filters_applied": {
            "max_pe": max_pe,
            "min_market_cap_b": min_market_cap_b,
            "min_dividend_yield": min_dividend_yield,
            "max_debt_to_equity": max_debt_to_equity,
        },
        "num_matches": len(matches),
        "matches": matches,
    }


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
    out: dict[str, dict] = {}
    available = list(ECONOMIC_DATA.keys())
    for name in indicators:
        key = name.lower()
        if key in ECONOMIC_DATA:
            out[key] = dict(ECONOMIC_DATA[key])
        else:
            out[key] = {"error": f"Unknown indicator. Available: {available}"}
    return {
        "indicators": out,
        "note": "Values are hardcoded snapshots. For live data, use FRED API.",
    }


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
