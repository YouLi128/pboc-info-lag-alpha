"""
Download ±2-day hourly CNH windows for every forward-guidance event.

Reads corpus_classified.csv, finds all event dates, downloads
4-day windows (event_date - 2 days to event_date + 2 days) concurrently,
merges all hours into one CSV.

Usage:
    python -m src.scraping.download_event_windows
    python -m src.scraping.download_event_windows --out data/raw/cnh_1h_all_events.csv
"""

from __future__ import annotations

import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from src.scraping.dukascopy_loader import fetch_hour, MAX_WORKERS, INSTRUMENT

logger = logging.getLogger(__name__)

# Each thread gets its own session — avoids connection-pool contention.
_thread_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (research-bot)"})
        _thread_local.session = s
    return _thread_local.session


def _fetch(instrument: str, dt: datetime) -> list[dict]:
    return fetch_hour(instrument, dt, _session())


def build_event_hours(classified_csv: str, pre_days: int = 2, post_days: int = 2) -> list[datetime]:
    """Return sorted list of unique UTC hour slots around all event dates."""
    df = pd.read_csv(classified_csv)
    df = df[df["segment_type"] == "forward_guidance"].copy()
    dates = pd.to_datetime(df["published"]).dt.date.unique()

    hour_set: set[datetime] = set()
    for d in dates:
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(days=pre_days)
        end   = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=post_days, hours=23)
        cur = start
        while cur <= end:
            hour_set.add(cur)
            cur += timedelta(hours=1)

    return sorted(hour_set)


def download_all(hours: list[datetime], workers: int = MAX_WORKERS) -> list[dict]:
    logger.info("Downloading %d unique hour slots with %d workers…", len(hours), workers)
    all_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, INSTRUMENT, h): h for h in hours}
        done = 0
        for fut in as_completed(futures):
            rows = fut.result()
            if rows:
                all_rows.extend(rows)
            done += 1
            if done % 500 == 0:
                logger.info("  %d / %d slots done  (%d ticks so far)",
                            done, len(hours), len(all_rows))

    return all_rows


def main(classified_csv: str, out: Path, workers: int = MAX_WORKERS) -> None:
    hours = build_event_hours(classified_csv)
    logger.info("%d unique hour slots across all event windows", len(hours))

    all_rows = download_all(hours, workers=workers)

    if not all_rows:
        logger.error("No data returned — check network or instrument name")
        return

    ticks = pd.DataFrame(all_rows)
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True)
    ticks = ticks.set_index("timestamp").sort_index()

    df = ticks["mid"].resample("1h").ohlc()
    df.columns = ["open", "high", "low", "close"]

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    logger.info("Saved %d hourly bars to %s", len(df), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified", default="data/processed/corpus_classified.csv")
    parser.add_argument("--out", type=Path, default=Path("data/raw/cnh_1h_all_events.csv"))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    main(args.classified, args.out, args.workers)
