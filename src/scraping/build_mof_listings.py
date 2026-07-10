"""
Build the MOF listings corpus (title/url/published stubs only) for the
2019-2025 window, mirroring build_ndrc_listings.py.

Usage:
    python -m src.scraping.build_mof_listings
    python -m src.scraping.build_mof_listings --max-pages 90 --since 2019-01-01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.scraping.mof_scraper import scrape_listings

logger = logging.getLogger(__name__)

OUT = Path("data/processed/mof_listings.csv")


def run(max_pages: int, since: str, out: Path) -> None:
    since_ts = pd.Timestamp(since)
    rows = []
    stop = False

    for article in scrape_listings(max_pages=max_pages):
        pub = pd.Timestamp(article["published"])
        if pub < since_ts:
            logger.info("Reached %s (< %s) — stopping.", pub.date(), since_ts.date())
            stop = True
            break
        rows.append(article)

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(
        "Saved %d listings (%s ~ %s) to %s%s",
        len(df), df["published"].min(), df["published"].max(), out,
        "" if stop else " (hit --max-pages before reaching --since date, consider raising it)",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--since", type=str, default="2019-01-01")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(max_pages=args.max_pages, since=args.since, out=args.out)
