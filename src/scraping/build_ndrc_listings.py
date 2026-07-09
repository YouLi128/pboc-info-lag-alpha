"""
Build the NDRC listings corpus (title/url/published stubs only, no full
text / no LLM) for the 2019-2025 window, mirroring how the PBOC
corpus_2019_2021_listings.csv / corpus_2022_2024_listings.csv files were
built. classify_ndrc_corpus.py consumes the output to do full-text fetch
+ LLM classification separately (so a scrape interruption doesn't waste
already-fetched listing pages).

Usage:
    python -m src.scraping.build_ndrc_listings
    python -m src.scraping.build_ndrc_listings --max-pages 45 --since 2019-01-01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.scraping.ndrc_scraper import scrape_listings

logger = logging.getLogger(__name__)

OUT = Path("data/processed/ndrc_listings.csv")


def run(max_pages: int, since: str, out: Path) -> None:
    since_ts = pd.Timestamp(since)
    rows = []
    stop = False

    for article in scrape_listings(max_pages=max_pages):
        pub = pd.Timestamp(article["published"].replace("/", "-"))
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
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--since", type=str, default="2019-01-01")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(max_pages=args.max_pages, since=args.since, out=args.out)
