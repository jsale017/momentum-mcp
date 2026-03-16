"""
Technical analysis module using pandas-ta.

Fetches historical data and computes RSI and MACD indicators,
returning the most recent values alongside price context.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pandas_ta as ta

from mcp_server.data import get_historical_data

logger = logging.getLogger(__name__)


async def analyze_technicals(
    ticker: str,
    period: str = "6mo",
) -> dict[str, Any]:
    """Compute RSI and MACD for a ticker and return the latest values.

    Fetches daily OHLCV data for the requested *period*, applies the
    **RSI(14)** and **MACD(12, 26, 9)** indicators via ``pandas-ta``,
    and returns the most recent readings together with price context.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        period: Lookback period for the underlying data. Defaults to
            ``"6mo"`` which provides enough bars for reliable indicator
            calculation. Accepted values match :func:`get_historical_data`.

    Returns:
        A dict with:

        - ``ticker`` — The requested symbol.
        - ``date`` — ISO-formatted date of the latest bar.
        - ``close`` — Most recent closing price.
        - ``rsi_14`` — Current RSI(14) value (0–100), or ``null``.
        - ``macd`` — MACD line value, or ``null``.
        - ``macd_signal`` — Signal line value, or ``null``.
        - ``macd_histogram`` — Histogram value, or ``null``.
        - ``analysis`` — A plain-English summary of the current readings.

    Raises:
        ValueError: If the ticker symbol is invalid or returns no data.
    """
    ticker = ticker.strip().upper()

    # Fetch OHLCV data (already validated inside data module)
    records = await get_historical_data(ticker, period=period, interval="1d")

    if len(records) < 35:
        raise ValueError(
            f"Insufficient data for '{ticker}': got {len(records)} bars but "
            f"need ≥35 for MACD(26) + signal(9). Try a longer period."
        )

    df = pd.DataFrame(records)
    df["close_float"] = pd.to_numeric(df["close"], errors="coerce")

    # --- RSI(14) ---
    rsi_series = ta.rsi(df["close_float"], length=14)

    # --- MACD(12, 26, 9) ---
    macd_df = ta.macd(df["close_float"], fast=12, slow=26, signal=9)

    # Extract the latest values
    latest = df.iloc[-1]
    latest_date = latest["date"]
    latest_close = latest["close"]

    rsi_val = _extract_last(rsi_series)
    macd_val = _extract_last(macd_df.iloc[:, 0]) if macd_df is not None else None
    signal_val = _extract_last(macd_df.iloc[:, 1]) if macd_df is not None else None
    hist_val = _extract_last(macd_df.iloc[:, 2]) if macd_df is not None else None

    # Build a plain-English summary
    analysis = _build_analysis(ticker, latest_close, rsi_val, macd_val, signal_val, hist_val)

    return {
        "ticker": ticker,
        "date": latest_date,
        "close": latest_close,
        "rsi_14": rsi_val,
        "macd": macd_val,
        "macd_signal": signal_val,
        "macd_histogram": hist_val,
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last(series: pd.Series | None) -> float | None:
    """Get the last non-NaN value from a series, rounded."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return round(float(val), 4)


def _build_analysis(
    ticker: str,
    close: Any,
    rsi: float | None,
    macd: float | None,
    signal: float | None,
    histogram: float | None,
) -> str:
    """Generate a concise plain-English analysis string."""
    parts: list[str] = [f"{ticker} last traded at {close}."]

    if rsi is not None:
        if rsi >= 70:
            parts.append(f"RSI(14) is {rsi:.1f} — overbought territory.")
        elif rsi <= 30:
            parts.append(f"RSI(14) is {rsi:.1f} — oversold territory.")
        else:
            parts.append(f"RSI(14) is {rsi:.1f} — neutral range.")

    if macd is not None and signal is not None:
        if macd > signal:
            parts.append("MACD is above the signal line (bullish crossover).")
        else:
            parts.append("MACD is below the signal line (bearish crossover).")

    if histogram is not None:
        direction = "expanding" if histogram > 0 else "contracting"
        parts.append(f"Histogram is {direction} at {histogram:.4f}.")

    return " ".join(parts)
