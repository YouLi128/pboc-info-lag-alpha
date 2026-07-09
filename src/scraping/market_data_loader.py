"""
Market data loaders.

CNH (offshore RMB):
    High-frequency CNH/USD data comes from Bloomberg or Wind manual exports.
    load_cnh_csv() reads the expected CSV format so downstream code is decoupled
    from the export format.  Fill in the actual column mapping once you have a
    sample export.

BTC (auxiliary 24/7 asset):
    fetch_btc_ohlcv() hits the public Binance REST klines endpoint — no API key
    required.  Used as a worked example and sanity-check for the signal pipeline.

CSI 300 (沪深300, broad-market extension branch):
    fetch_csi300_daily() pulls daily OHLCV via the akshare wrapper around
    Sina's index feed — no API key required, back to 2002.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import akshare as ak
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CNH placeholder loader
# ---------------------------------------------------------------------------

# Expected CSV columns — update once you have a real Bloomberg/Wind export.
# Bloomberg typically exports: Date, Time, Open, High, Low, Close, Volume
# Wind typically exports: 时间, 开盘价, 最高价, 最低价, 收盘价, 成交量
CNH_COLUMN_MAP = {
    # raw column name → canonical name
    "Date": "date",
    "Time": "time",
    "Close": "close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Volume": "volume",
}


def load_cnh_csv(path: str) -> pd.DataFrame:
    """
    Load a Bloomberg/Wind manual export and return a tidy DataFrame.

    Expected output schema:
        timestamp (DatetimeIndex, UTC)
        open, high, low, close (float, CNH per USD)
        volume (float, optional)

    TODO: Adjust CNH_COLUMN_MAP and the timestamp assembly logic to match your
    actual export format.  Bloomberg and Wind use different date/time layouts.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in CNH_COLUMN_MAP.items() if k in df.columns})

    # TODO: combine date + time columns if they are separate; parse timezone.
    # Example (Bloomberg intraday):
    #   df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], utc=True)
    # Example (Wind):
    #   df["timestamp"] = pd.to_datetime(df["时间"], utc=False).dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC")

    if "timestamp" not in df.columns:
        logger.warning(
            "CNH CSV missing 'timestamp' column — update load_cnh_csv() for your export format."
        )
        df["timestamp"] = pd.NaT

    df = df.set_index("timestamp").sort_index()
    logger.info("Loaded %d CNH rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# BTC loader via Binance REST (public, no auth required)
# ---------------------------------------------------------------------------

BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = "/api/v3/klines"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_btc_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance and return a tidy DataFrame.

    Args:
        symbol:   Binance trading pair, default "BTCUSDT".
        interval: Kline interval string ("1m", "5m", "1h", "1d", …).
        start_dt: Inclusive start (UTC).  Defaults to `limit` bars before now.
        end_dt:   Exclusive end (UTC).  Defaults to now.
        limit:    Max rows per API call (Binance cap: 1000).

    Returns:
        DataFrame indexed by open_time (UTC DatetimeIndex) with columns:
        open, high, low, close (float), volume (float).
    """
    params: dict = {"symbol": symbol, "interval": interval, "limit": limit}

    if start_dt is not None:
        params["startTime"] = int(start_dt.timestamp() * 1000)
    if end_dt is not None:
        params["endTime"] = int(end_dt.timestamp() * 1000)

    url = f"{BINANCE_BASE}{BINANCE_KLINES}"
    logger.info("Fetching Binance klines: %s %s", symbol, interval)
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json(), columns=KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    logger.info("Fetched %d %s/%s klines from Binance", len(df), symbol, interval)
    return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# CSI 300 loader via akshare (Sina index feed, no auth required)
# ---------------------------------------------------------------------------

CSI300_SYMBOL = "sh000300"


def fetch_csi300_daily(symbol: str = CSI300_SYMBOL) -> pd.DataFrame:
    """
    Fetch full daily OHLCV history for the CSI 300 index.

    Returns:
        DataFrame indexed by date (tz-naive) with columns
        open, high, low, close (float), volume (float).
    """
    logger.info("Fetching CSI 300 daily history (%s)…", symbol)
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    logger.info("Fetched %d CSI 300 daily rows (%s ~ %s)", len(df), df.index.min(), df.index.max())
    return df[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch_btc_ohlcv(interval="1h", limit=5)
    print(df)
    csi300 = fetch_csi300_daily()
    print(csi300.tail())
