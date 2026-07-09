"""
Robust hourly CNH downloader for all event windows (2019-2025).

Strategy to avoid Dukascopy rate-limiting:
  - 5 concurrent workers (not 20)
  - Batches of 100 hour-slots with an 8-second pause between batches
  - Downloads year by year, saves per-year CSV as checkpoint
  - Skips years whose CSV already exists (resume-capable)
  - Merges all years at the end

Usage:
    python -m src.scraping.download_hourly_robust
    python -m src.scraping.download_hourly_robust --force   # re-download everything
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.scraping.dukascopy_loader import fetch_hour, INSTRUMENT

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
WORKERS    = 5
PAUSE_SEC  = 8    # between batches
OUT_DIR    = Path("data/raw/hourly_by_year")
FINAL_OUT  = Path("data/raw/cnh_1h_all_events.csv")
CLASSIFIED = "data/processed/corpus_classified.csv"

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        _local.s = s
    return _local.s


def _fetch(dt: datetime) -> list[dict]:
    return fetch_hour(INSTRUMENT, dt, _session())


def event_hours_for_year(year: int, pre: int = 2, post: int = 2) -> list[datetime]:
    """All unique UTC hour slots around forward-guidance events in a given year."""
    df = pd.read_csv(CLASSIFIED)
    df = df[df["segment_type"] == "forward_guidance"].copy()
    dates = pd.to_datetime(df["published"]).dt.date.unique()
    dates = [d for d in dates if d.year == year]

    hour_set: set[datetime] = set()
    for d in dates:
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(days=pre)
        end   = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=post, hours=23)
        cur = start
        while cur <= end:
            hour_set.add(cur)
            cur += timedelta(hours=1)
    return sorted(hour_set)


def download_year(year: int, out_path: Path) -> int:
    """Download all event-window hours for one year. Returns tick count."""
    hours = event_hours_for_year(year)
    if not hours:
        logger.info("Year %d: no events, skipping.", year)
        return 0

    logger.info("Year %d: %d hour slots → %s", year, len(hours), out_path)
    all_rows: list[dict] = []
    n_batches = (len(hours) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(n_batches):
        batch = hours[b_idx * BATCH_SIZE:(b_idx + 1) * BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futs = [pool.submit(_fetch, h) for h in batch]
            for f in as_completed(futs):
                rows = f.result()
                if rows:
                    all_rows.extend(rows)

        logger.info("  batch %d/%d done | ticks so far: %d",
                    b_idx + 1, n_batches, len(all_rows))

        if b_idx + 1 < n_batches:
            time.sleep(PAUSE_SEC)

    if not all_rows:
        logger.warning("Year %d: no ticks returned (weekend-only events or blocked).", year)
        return 0

    ticks = pd.DataFrame(all_rows)
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)
    ticks = ticks.set_index("timestamp").sort_index()
    df = ticks["mid"].resample("1h").ohlc()
    df.columns = ["open", "high", "low", "close"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    logger.info("Year %d: saved %d hourly bars to %s", year, len(df), out_path)
    return len(all_rows)


def merge_years(years: list[int]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = OUT_DIR / f"cnh_1h_{y}.csv"
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if len(df):
                parts.append(df)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged


def main(force: bool = False) -> None:
    years = list(range(2019, 2026))
    logger.info("Downloading event-window hourly data for years: %s", years)

    for year in years:
        out_path = OUT_DIR / f"cnh_1h_{year}.csv"
        if out_path.exists() and not force:
            logger.info("Year %d: already exists (%d rows), skipping.",
                        year, len(pd.read_csv(out_path)))
            continue
        download_year(year, out_path)
        # Extra cooldown between years
        logger.info("Cooling down 30s before next year…")
        time.sleep(30)

    logger.info("Merging all years → %s", FINAL_OUT)
    merged = merge_years(years)
    if merged.empty:
        logger.error("No data to merge.")
        return

    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FINAL_OUT)
    logger.info("Done. %d hourly bars saved to %s  (%s ~ %s)",
                len(merged), FINAL_OUT, merged.index[0], merged.index[-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if year file exists")
    args = parser.parse_args()
    main(force=args.force)
