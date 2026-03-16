"""
Market data module using yfinance.

Provides clean OHLCV data retrieval with ISO-formatted timestamps,
wrapped in async for non-blocking MCP tool calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Valid periods and intervals accepted by yfinance
VALID_PERIODS = {
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
}
VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo",
}


async def get_historical_data(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
) -> list[dict[str, Any]]:
    """Fetch OHLCV historical data for a ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``, ``"MSFT"``).
        period: Lookback period. One of: ``1d``, ``5d``, ``1mo``, ``3mo``,
            ``6mo``, ``1y``, ``2y``, ``5y``, ``10y``, ``ytd``, ``max``.
            Defaults to ``"3mo"``.
        interval: Bar interval. One of: ``1m``, ``2m``, ``5m``, ``15m``,
            ``30m``, ``60m``, ``90m``, ``1h``, ``1d``, ``5d``, ``1wk``,
            ``1mo``, ``3mo``. Defaults to ``"1d"``.

    Returns:
        A list of dicts with keys: ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume``. Dates are ISO-8601 formatted strings.

    Raises:
        ValueError: If *period* or *interval* is invalid, or the ticker
            returns no data.
    """
    ticker = ticker.strip().upper()

    if period not in VALID_PERIODS:
        raise ValueError(
            f"Invalid period '{period}'. Must be one of: {sorted(VALID_PERIODS)}"
        )
    if interval not in VALID_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. Must be one of: {sorted(VALID_INTERVALS)}"
        )

    def _download() -> pd.DataFrame:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception as exc:
            logger.error("yfinance download failed for %s: %s", ticker, exc)
            raise ValueError(
                f"Failed to fetch data for '{ticker}'. "
                f"The ticker may be invalid or Yahoo Finance may be rate-limiting."
            ) from exc

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for ticker '{ticker}' "
                f"(period={period}, interval={interval}). "
                f"Verify the ticker symbol is correct."
            )
        return df

    df = await asyncio.to_thread(_download)

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Build clean JSON-serialisable output
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        date_str = (
            idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
        )
        records.append(
            {
                "date": date_str,
                "open": _round(row.get("Open")),
                "high": _round(row.get("High")),
                "low": _round(row.get("Low")),
                "close": _round(row.get("Close")),
                "volume": int(row.get("Volume", 0)),
            }
        )

    logger.info(
        "Fetched %d bars for %s (period=%s, interval=%s)",
        len(records), ticker, period, interval,
    )
    return records


def _round(value: Any, decimals: int = 4) -> float | None:
    """Round a numeric value, returning None for NaN / missing."""
    try:
        f = float(value)
        if pd.isna(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return None
