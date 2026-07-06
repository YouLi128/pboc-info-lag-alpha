"""
Dukascopy free historical FX data loader.

Downloads USDCNH hourly OHLCV data from Dukascopy's public HTTP endpoint.
No account or API key required.

Data is stored as bi5 (LZMA-compressed binary) files, one per hour.
Each record is 20 bytes: timestamp(4) + ask_open/high/low/close(4×4) + volume(4).
Prices are in integer pips (divide by 100000 to get actual rate).

Usage:
    python -m src.scraping.dukascopy_loader --start 2022-01-01 --end 2024-12-31
    python -m src.scraping.dukascopy_loader --start 2022-01-01 --end 2024-12-31 --interval h1
"""

from __future__ import annotations

import argparse
import logging
import lzma
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
INSTRUMENT = "USDCNH"
PRICE_DIVISOR = 100_000.0
# bi5 tick record: uint32 ms_offset, uint32 ask, uint32 bid, float32 ask_vol, float32 bid_vol
RECORD_FORMAT = ">IIIff"
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)  # 20 bytes


def _bi5_url(instrument: str, dt: datetime) -> str:
    return (
        f"{BASE_URL}/{instrument}/"
        f"{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/"
        f"{dt.hour:02d}h_ticks.bi5"
    )


def _parse_bi5(data: bytes, hour_dt: datetime) -> list[dict]:
    """Decompress and parse a single bi5 hour file into tick rows."""
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        return []

    n = len(raw) // RECORD_SIZE
    rows = []
    for i in range(n):
        chunk = raw[i * RECORD_SIZE: (i + 1) * RECORD_SIZE]
        ms_offset, ask, bid, ask_vol, bid_vol = struct.unpack(RECORD_FORMAT, chunk)
        mid = (ask + bid) / 2 / PRICE_DIVISOR
        ts = hour_dt + timedelta(milliseconds=ms_offset)
        rows.append({
            "timestamp": ts,
            "ask":  ask / PRICE_DIVISOR,
            "bid":  bid / PRICE_DIVISOR,
            "mid":  mid,
            "ask_vol": ask_vol,
            "bid_vol": bid_vol,
        })
    return rows


def fetch_hour(instrument: str, dt: datetime, session: requests.Session) -> list[dict]:
    url = _bi5_url(instrument, dt)
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return _parse_bi5(r.content, dt)
    except Exception as exc:
        logger.debug("Skip %s: %s", url, exc)
        return []


def fetch_range(
    start: datetime,
    end: datetime,
    instrument: str = INSTRUMENT,
    interval: str = "h1",
    out_path: Path | None = None,
) -> pd.DataFrame:
    """
    Download and return OHLCV data for the given UTC date range.

    Args:
        start:      Start datetime (UTC).
        end:        End datetime (UTC, inclusive).
        instrument: Dukascopy instrument code, default "USDCNH".
        interval:   "tick" for raw ticks, "h1" to resample to 1-hour bars.
        out_path:   If provided, save CSV here.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research-bot)"})

    all_rows: list[dict] = []
    current = start.replace(minute=0, second=0, microsecond=0)
    total_hours = int((end - start).total_seconds() / 3600) + 1

    logger.info("Fetching %d hours of %s data …", total_hours, instrument)

    fetched = 0
    while current <= end:
        rows = fetch_hour(instrument, current, session)
        all_rows.extend(rows)
        fetched += 1

        if fetched % 100 == 0:
            logger.info(
                "  %d / %d hours  (%s)", fetched, total_hours, current.strftime("%Y-%m-%d")
            )

        current += timedelta(hours=1)
        time.sleep(0.05)  # polite delay

    if not all_rows:
        logger.warning("No data returned — check instrument name or date range.")
        return pd.DataFrame()

    ticks = pd.DataFrame(all_rows)
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)
    ticks = ticks.set_index("timestamp").sort_index()

    if interval == "h1":
        df = ticks["mid"].resample("1h").ohlc()
        df.columns = ["open", "high", "low", "close"]
    else:
        df = ticks

    logger.info("Done. %d rows  (%s ~ %s)", len(df), df.index[0], df.index[-1])

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
        logger.info("Saved to %s", out_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--interval", choices=["tick", "h1"], default="h1")
    parser.add_argument("--out", type=Path,
                        default=Path("data/raw/cnh_1h_2022_2024.csv"))
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    fetch_range(start_dt, end_dt, interval=args.interval, out_path=args.out)
