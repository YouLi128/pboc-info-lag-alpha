"""
Pilot: measure English-language news lag behind PBOC Chinese releases via GDELT.

For each PBOC forward_guidance event, query the free GDELT 2.0 DOC API for
English-language articles mentioning PBOC monetary policy within a window
starting at the Chinese release and extending a few days forward. The
timestamp of the earliest matching English article, minus the Chinese
release timestamp, is a proxy for "translation lag" — this is the missing
comparison needed to actually test the H1 cross-border information-lag
claim, rather than just testing "does the market react to policy news".

GDELT enforces ~1 request per 5s and is intermittently flaky from this
network, so this pilot runs a small sample (~15-20 events) sequentially
with generous spacing before committing to the full 205-event run.

Usage:
    python -m src.scraping.gdelt_lag_pilot --events /tmp/gdelt_pilot_events.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '(PBOC OR "People\'s Bank of China") (rate OR reserve OR RRR OR monetary OR liquidity) sourcelang:english'
WINDOW_DAYS = 4
REQUEST_GAP = 6  # seconds between requests, GDELT asks for >=5s


def query_gdelt(start: datetime, end: datetime, retries: int = 3) -> list[dict]:
    params = {
        "query": QUERY,
        "mode": "artlist",
        "maxrecords": 25,
        "format": "json",
        "sort": "dateasc",  # earliest first — must not be datedesc or a busy window truncates the earliest articles
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for attempt in range(retries):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=25)
            if r.status_code == 429:
                logger.debug("429, backing off")
                time.sleep(8 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("articles", [])
        except Exception as exc:
            logger.debug("GDELT query failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    return []


def main(events_csv: str) -> None:
    events = pd.read_csv(events_csv, parse_dates=["release_utc"])
    rows = []

    for i, ev in events.iterrows():
        t0 = ev["release_utc"]
        window_end = t0 + timedelta(days=WINDOW_DAYS)
        logger.info("[%d/%d] %s -> querying GDELT %s ~ %s",
                    i + 1, len(events), ev["event_date"], t0.date(), window_end.date())

        articles = query_gdelt(t0, window_end)
        if articles:
            dates = sorted(a["seendate"] for a in articles if a.get("seendate"))
            earliest = datetime.strptime(dates[0], "%Y%m%dT%H%M%SZ") if dates else None
            lag_hours = (earliest - t0.tz_localize(None)).total_seconds() / 3600 if earliest else None
        else:
            earliest, lag_hours = None, None

        rows.append({
            "event_date": ev["event_date"],
            "release_utc": t0,
            "n_english_articles": len(articles),
            "earliest_english_utc": earliest,
            "lag_hours": lag_hours,
        })
        if earliest:
            logger.info("  -> %d English articles found, earliest: %s (lag %.1fh)",
                        len(articles), earliest, lag_hours)
        else:
            logger.info("  -> %d English articles found, none in window", len(articles))

        time.sleep(REQUEST_GAP)

    out = pd.DataFrame(rows)
    out.to_csv("data/processed/gdelt_lag_pilot.csv", index=False)
    print(out.to_string())
    print(f"\n{out['n_english_articles'].gt(0).sum()} / {len(out)} events had at least one English article in window")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="/tmp/gdelt_pilot_events.csv")
    args = parser.parse_args()
    main(args.events)
