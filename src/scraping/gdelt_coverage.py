"""
Full-sample GDELT English-coverage check for all 205 PBOC forward_guidance
events — the H2 test this project actually needs: not "how many hours late
was English coverage" (pilot showed it's ~0-2.5h when it exists at all —
see gdelt_lag_pilot.py), but "did this event get English-language wire
coverage at all". Events with NO English coverage are the ones where CNH/
repo-rate markets can only be responding to the Chinese-language original —
the cleanest available test of the cross-border information-lag hypothesis.

Usage:
    python -m src.scraping.gdelt_coverage
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from src.analysis.event_study_pipeline import build_events

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '(PBOC OR "People\'s Bank of China") (rate OR reserve OR RRR OR monetary OR liquidity) sourcelang:english'
WINDOW_DAYS = 4
REQUEST_GAP = 6
OUT = "data/processed/gdelt_coverage.csv"


def query_gdelt(start: datetime, end: datetime, query: str = QUERY, retries: int = 3) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 25,
        "format": "json",
        "sort": "dateasc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for attempt in range(retries):
        try:
            r = requests.get(GDELT_URL, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json().get("articles", [])
        except Exception as exc:
            logger.debug("GDELT query failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(5 * (attempt + 1))
    return []


def load_done(out_path: str) -> set:
    try:
        df = pd.read_csv(out_path, parse_dates=["release_utc"])
        return set(df["release_utc"].astype(str))
    except FileNotFoundError:
        return set()


def main(classified_csv: str = "data/processed/corpus_classified.csv",
        out_path: str = OUT, query: str = QUERY) -> None:
    events = build_events(classified_csv)
    done = load_done(out_path)
    todo = events[~events["release_utc"].astype(str).isin(done)]
    logger.info("Total events: %d | already done: %d | remaining: %d", len(events), len(done), len(todo))

    write_header = not done
    mode = "w" if write_header else "a"

    with open(out_path, mode, encoding="utf-8") as f:
        if write_header:
            f.write("event_date,release_utc,n_english_articles,earliest_english_utc,lag_hours,covered\n")

        for i, (_, ev) in enumerate(todo.iterrows()):
            t0 = ev["release_utc"]
            window_end = t0 + timedelta(days=WINDOW_DAYS)
            articles = query_gdelt(t0, window_end, query=query)

            if articles:
                dates = sorted(a["seendate"] for a in articles if a.get("seendate"))
                earliest = datetime.strptime(dates[0], "%Y%m%dT%H%M%SZ") if dates else None
                lag_hours = (earliest - t0.tz_localize(None)).total_seconds() / 3600 if earliest else None
            else:
                earliest, lag_hours = None, None

            covered = int(bool(articles))
            f.write(f"{ev['event_date']},{t0},{len(articles)},{earliest or ''},{lag_hours if lag_hours is not None else ''},{covered}\n")
            f.flush()

            if (i + 1) % 20 == 0:
                logger.info("[%d/%d] %s | covered=%s", i + 1, len(todo), ev["event_date"], bool(articles))

            time.sleep(REQUEST_GAP)

    logger.info("Done. Saved to %s", out_path)
    final = pd.read_csv(out_path)
    logger.info("Coverage rate: %d / %d (%.0f%%)", final["covered"].sum(), len(final), 100 * final["covered"].mean())


NDRC_QUERY = '("NDRC" OR "National Development and Reform Commission" OR "China\'s economic planner") sourcelang:english'

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified", default="data/processed/corpus_classified.csv")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--query", default=QUERY)
    args = parser.parse_args()
    main(classified_csv=args.classified, out_path=args.out, query=args.query)
