"""
Generic full-text re-fetcher for already-classified forward_guidance
articles — parallels fetch_pboc_fulltext.py but works for any source
whose scraper module exposes fetch_article_text(url).

classify_*_corpus.py scripts never persist raw article text (only
title/url/published/segment_type/stance/confidence/reasoning), so
building a dictionary or continuous-score signal requires re-fetching.
No LLM calls here — pure HTTP scraping.

Usage:
    python -m src.scraping.fetch_fulltext --source ndrc --urls /tmp/ndrc_fg_urls.csv --out data/processed/ndrc_fg_fulltext.csv
    python -m src.scraping.fetch_fulltext --source mof --urls /tmp/mof_fg_urls.csv --out data/processed/mof_fg_fulltext.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCRAPER_MODULES = {
    "ndrc": "src.scraping.ndrc_scraper",
    "mof": "src.scraping.mof_scraper",
}


def load_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    df = pd.read_csv(out, usecols=["url"])
    return set(df["url"].dropna())


def run(source: str, urls_csv: str, out: Path) -> None:
    fetch_article_text = importlib.import_module(SCRAPER_MODULES[source]).fetch_article_text

    listings = pd.read_csv(urls_csv)
    done = load_done(out)
    todo = listings[~listings["url"].isin(done)]
    logger.info("[%s] Total: %d | Done: %d | Remaining: %d", source, len(listings), len(done), len(todo))

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
                logger.info("[%s] [%d / %d]", source, i + 1, len(todo))

    logger.info("[%s] Done. Output: %s", source, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=list(SCRAPER_MODULES), required=True)
    parser.add_argument("--urls", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.urls, args.out)
