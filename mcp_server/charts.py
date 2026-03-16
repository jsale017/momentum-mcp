"""
Chart generation module using mplfinance.

Renders candlestick + volume charts as static PNGs and returns both the
file path and a base64-encoded string for inline display.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless rendering — must be set before importing pyplot

import mplfinance as mpf  # noqa: E402
import pandas as pd  # noqa: E402

from mcp_server.data import get_historical_data  # noqa: E402

logger = logging.getLogger(__name__)

# Default output directory for saved charts
CHARTS_DIR = Path("./charts")

# Clean dark style for chart rendering
_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#22c55e",
        down="#ef4444",
        wick={"up": "#22c55e", "down": "#ef4444"},
        edge={"up": "#22c55e", "down": "#ef4444"},
        volume={"up": "#22c55e80", "down": "#ef444480"},
    ),
    facecolor="#0f0f0f",
    figcolor="#0f0f0f",
    gridcolor="#1a1a1a",
    gridstyle="--",
    y_on_right=True,
    rc={
        "font.size": 9,
        "axes.labelcolor": "#cccccc",
        "xtick.color": "#888888",
        "ytick.color": "#888888",
    },
)


async def generate_chart(
    ticker: str,
    period: str = "3mo",
    interval: str = "1d",
    style: str = "dark",
) -> dict[str, str]:
    """Generate a candlestick chart for a ticker symbol.

    Fetches OHLCV data, renders a candlestick chart with volume panel
    using ``mplfinance``, saves the PNG to the ``./charts/`` directory,
    and returns both the file path and a base64-encoded representation.

    Args:
        ticker: Stock ticker symbol (e.g. ``"AAPL"``).
        period: Lookback period (e.g. ``"3mo"``, ``"1y"``).
            Defaults to ``"3mo"``.
        interval: Bar interval (e.g. ``"1d"``, ``"1h"``).
            Defaults to ``"1d"``.
        style: Chart colour theme. Currently only ``"dark"`` is
            supported. Reserved for future expansion.

    Returns:
        A dict with:

        - ``ticker`` — The symbol charted.
        - ``period`` — The period used.
        - ``interval`` — The interval used.
        - ``bars`` — Number of bars rendered.
        - ``path`` — Absolute path to the saved PNG file.
        - ``base64`` — Base64-encoded PNG string (UTF-8).

    Raises:
        ValueError: If the ticker is invalid or returns no data.
    """
    ticker = ticker.strip().upper()

    # Fetch data
    records = await get_historical_data(ticker, period=period, interval=interval)

    if len(records) < 5:
        raise ValueError(
            f"Not enough data to chart '{ticker}': got {len(records)} bars."
        )

    # Build DataFrame in mplfinance-expected format
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )

    # Coerce to numeric (yfinance occasionally returns strings)
    for col_name in ("Open", "High", "Low", "Close", "Volume"):
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

    # Render in a background thread (matplotlib is not thread-safe by
    # default, but Agg backend + isolated figure is fine here)
    def _render() -> tuple[str, str]:
        # Save to file
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{ticker}_{period}_{interval}.png"
        filepath = CHARTS_DIR / filename

        mpf.plot(
            df,
            type="candle",
            style=_STYLE,
            volume=True,
            title=f"\n{ticker}  ({period} / {interval})",
            figsize=(12, 7),
            savefig=dict(fname=str(filepath), dpi=150, bbox_inches="tight"),
        )

        # Also render to an in-memory buffer for base64
        buf = io.BytesIO()
        mpf.plot(
            df,
            type="candle",
            style=_STYLE,
            volume=True,
            title=f"\n{ticker}  ({period} / {interval})",
            figsize=(12, 7),
            savefig=dict(fname=buf, dpi=150, bbox_inches="tight"),
        )
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")

        return str(filepath.resolve()), b64

    path, b64_str = await asyncio.to_thread(_render)

    logger.info("Chart saved: %s (%d bars)", path, len(df))

    return {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "bars": len(df),
        "path": path,
        "base64": b64_str,
    }
