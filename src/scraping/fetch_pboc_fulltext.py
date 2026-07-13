"""
Re-fetch full text for already-classified PBOC forward_guidance articles.

classify_corpus.py never persisted raw article text (only text_length),
so building a keyword-dictionary baseline requires re-fetching. No LLM
calls here — pure HTTP scraping, reusing pboc_scraper.fetch_article_text.

Usage:
    python -m src.scraping.fetch_pboc_fulltext
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import pandas as pd

from src.scraping.pboc_scraper import fetch_article_text

logger = logging.getLogger(__name__)


def load_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    df = pd.read_csv(out, usecols=["url"])
    return set(df["url"].dropna())


def run(urls_csv: str, out: Path) -> None:
    listings = pd.read_csv(urls_csv)
    done = load_done(out)
    todo = listings[~listings["url"].isin(done)]
    logger.info("Total: %d | Done: %d | Remaining: %d", len(listings), len(done), len(todo))

    write_header = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "published", "text"])
        if write_header:
            writer.writeheader()

        for i, (_, row) in enumerate(todo.iterrows()):
            try:
                text = fetch_article_text(row["url"])
            except Exception as e:
                logger.warning("[%d] fetch failed: %s", i + 1, e)
                text = ""
            writer.writerow({"url": row["url"], "published": row["published"], "text": text})
            f.flush()
            if (i + 1) % 20 == 0:
                logger.info("[%d / %d]", i + 1, len(todo))

    logger.info("Done. Output: %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default="/tmp/pboc_fg_urls.csv")
    parser.add_argument("--out", type=Path, default=Path("data/processed/pboc_fg_fulltext.csv"))
    args = parser.parse_args()
    run(args.urls, args.out)
