"""
Stock screening module using tradingview-screener.

Provides preset-based stock screens that map to TradingView's scanner API,
returning filtered results with key metrics.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tradingview_screener import Query, col

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset filter definitions
# ---------------------------------------------------------------------------
# Each preset is a callable that receives a Query and returns a filtered Query.

PRESET_FILTERS: dict[str, Any] = {
    "most_active": lambda q: q.order_by("volume", ascending=False),
    "new_highs": lambda q: q.where(col("High.All") == True),  # noqa: E712
    "new_lows": lambda q: q.where(col("Low.All") == True),  # noqa: E712
    "overbought": lambda q: q.where(col("RSI") > 70).order_by("RSI", ascending=False),
    "oversold": lambda q: q.where(col("RSI") < 30).order_by("RSI", ascending=True),
    "high_relative_volume": lambda q: q.where(
        col("relative_volume_10d_calc") > 2.0
    ).order_by("relative_volume_10d_calc", ascending=False),
}

# Columns we always request from the screener
_DEFAULT_COLUMNS = [
    "name",
    "close",
    "volume",
    "change",
    "relative_volume_10d_calc",
    "RSI",
    "market_cap_basic",
]


async def run_stock_screen(
    preset: str = "most_active",
    market: str = "america",
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Run a stock screen using TradingView's scanner API.

    Args:
        preset: Screen preset to use. One of:
            ``most_active``, ``new_highs``, ``new_lows``,
            ``overbought``, ``oversold``, ``high_relative_volume``.
        market: Market to scan. Defaults to ``"america"``.
        limit: Maximum number of results to return (1–100).

    Returns:
        A list of dicts, each containing:
        ``ticker``, ``close``, ``volume``, ``change_pct``,
        ``relative_volume``, ``rsi``, ``market_cap``.

    Raises:
        ValueError: If *preset* is not recognised.
    """
    if preset not in PRESET_FILTERS:
        available = ", ".join(sorted(PRESET_FILTERS.keys()))
        raise ValueError(
            f"Unknown preset '{preset}'. Available presets: {available}"
        )

    limit = max(1, min(limit, 100))

    def _execute() -> list[dict[str, Any]]:
        query = Query().select(*_DEFAULT_COLUMNS).limit(limit)

        # Apply the market filter
        query = query.set_markets(market)

        # Apply preset-specific filters
        query = PRESET_FILTERS[preset](query)

        try:
            _, raw_rows = query.get_scanner_data()
        except Exception as exc:
            logger.error("Screener query failed for preset '%s': %s", preset, exc)
            return []

        if raw_rows is None or raw_rows.empty:
            logger.info("Screener returned 0 results for preset '%s'", preset)
            return []

        results: list[dict[str, Any]] = []
        for _, row in raw_rows.iterrows():
            ticker_raw = row.get("name", "")
            # TradingView returns tickers as "EXCHANGE:SYMBOL" — strip the exchange
            ticker = ticker_raw.split(":")[-1] if ":" in str(ticker_raw) else str(ticker_raw)

            results.append(
                {
                    "ticker": ticker,
                    "close": _safe_float(row.get("close")),
                    "volume": _safe_int(row.get("volume")),
                    "change_pct": _safe_float(row.get("change")),
                    "relative_volume": _safe_float(row.get("relative_volume_10d_calc")),
                    "rsi": _safe_float(row.get("RSI")),
                    "market_cap": _safe_float(row.get("market_cap_basic")),
                }
            )

        return results

    return await asyncio.to_thread(_execute)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert a value to int, returning None on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
