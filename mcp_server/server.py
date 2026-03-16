"""
momentum-mcp: Main MCP server entry point.

Registers all quantitative trading tools with FastMCP and exposes them
for AI agent consumption via the Model Context Protocol.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from mcp_server.screener import run_stock_screen as _run_stock_screen
from mcp_server.data import get_historical_data as _get_historical_data
from mcp_server.technicals import analyze_technicals as _analyze_technicals
from mcp_server.charts import generate_chart as _generate_chart
from mcp_server.news import (
    fetch_ticker_news as _fetch_ticker_news,
    extract_article_text as _extract_article_text,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "momentum",
    instructions=(
        "Quantitative trading analysis server. Use these tools to screen "
        "stocks, fetch OHLCV data, compute technical indicators (RSI, MACD), "
        "generate candlestick charts, and retrieve financial news articles."
    ),
)


# ═══════════════════════════════════════════════════════════════════════════
# Tool: run_stock_screen
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def run_stock_screen(
    preset: str = "most_active",
    market: str = "america",
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Run a stock screen using TradingView's scanner.

    Scans the market for stocks matching a preset filter and returns
    a ranked list with key metrics.

    Args:
        preset: The screen to run. Options:
            - ``most_active`` — highest volume stocks
            - ``new_highs`` — stocks hitting all-time highs
            - ``new_lows`` — stocks hitting all-time lows
            - ``overbought`` — RSI > 70
            - ``oversold`` — RSI < 30
            - ``high_relative_volume`` — relative volume > 2x the 10-day average
        market: Market to scan (default: ``"america"``).
        limit: Max results to return, 1–100 (default: 25).

    Returns:
        List of dicts with: ticker, close, volume, change_pct,
        relative_volume, rsi, market_cap.
    """
    return await _run_stock_screen(preset=preset, market=market, limit=limit)


# ═══════════════════════════════════════════════════════════════════════════
# Tool: get_historical_data
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_historical_data(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> list[dict[str, Any]]:
    """Fetch OHLCV historical price data for a stock.

    Returns clean, JSON-serialisable candlestick data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``, ``"TSLA"``).
        period: Lookback period. Options: ``1d``, ``5d``, ``1mo``, ``3mo``,
            ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``, ``ytd``, ``max``.
        interval: Candle interval. Options: ``1m``, ``5m``, ``15m``, ``30m``,
            ``1h``, ``1d``, ``1wk``, ``1mo``.

    Returns:
        List of dicts with: date (ISO-8601), open, high, low, close, volume.
    """
    return await _get_historical_data(ticker=ticker, period=period, interval=interval)


# ═══════════════════════════════════════════════════════════════════════════
# Tool: analyze_technicals
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def analyze_technicals(
    ticker: str,
    period: str = "6mo",
) -> dict[str, Any]:
    """Compute RSI and MACD technical indicators for a stock.

    Fetches daily OHLCV data, applies RSI(14) and MACD(12, 26, 9), and
    returns the latest readings with a plain-English analysis summary.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        period: Lookback period for indicator calculation. Use at least
            ``"3mo"`` for reliable results. Default: ``"6mo"``.

    Returns:
        Dict with: ticker, date, close, rsi_14, macd, macd_signal,
        macd_histogram, analysis.
    """
    return await _analyze_technicals(ticker=ticker, period=period)


# ═══════════════════════════════════════════════════════════════════════════
# Tool: generate_chart
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def generate_chart(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    style: str = "dark",
    show_emas: bool = True,
) -> dict[str, str]:
    """Generate a candlestick chart with EMA overlays for a stock.

    Renders a professional candlestick chart with volume panel,
    stacked EMA overlays (8/21/34/55/89), and saves it as a PNG.
    Returns both the file path and a base64-encoded string.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        period: Lookback period (e.g. ``"6mo"``, ``"1y"``).
        interval: Candle interval (e.g. ``"1d"``, ``"1h"``).
        style: Chart theme. Currently ``"dark"`` (default).
        show_emas: Overlay EMA stack (8/21/34/55/89). Default: ``True``.

    Returns:
        Dict with: ticker, period, interval, bars, emas, path, base64.
    """
    return await _generate_chart(
        ticker=ticker, period=period, interval=interval,
        style=style, show_emas=show_emas,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tool: fetch_ticker_news
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def fetch_ticker_news(
    ticker: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Fetch recent financial news headlines for a stock.

    Queries Yahoo Finance and Google News RSS feeds, de-duplicates by URL,
    and returns the most recent headlines.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        limit: Max headlines to return, 1–50 (default: 10).

    Returns:
        List of dicts with: title, url, published, source.
    """
    return await _fetch_ticker_news(ticker=ticker, limit=limit)


# ═══════════════════════════════════════════════════════════════════════════
# Tool: extract_article_text
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def extract_article_text(
    url: str,
) -> dict[str, Any]:
    """Extract the full-text body of a news article.

    Downloads the page at the given URL and uses trafilatura to strip
    ads, navigation, and boilerplate, returning the clean article text.

    Args:
        url: Full URL of the article to extract
            (e.g. ``"https://finance.yahoo.com/news/..."``).

    Returns:
        Dict with: url, title, text, word_count, error (null if success).
    """
    return await _extract_article_text(url=url)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting momentum MCP server...")
    mcp.run()
