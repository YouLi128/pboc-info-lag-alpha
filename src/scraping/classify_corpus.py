"""
Classify all 2022-2024 PBOC articles with LLM.

Supports resume: skips URLs already present in the output CSV.

Usage:
    python -m src.scraping.classify_corpus
    python -m src.scraping.classify_corpus --out data/processed/corpus_classified.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.llm_processing.stance_classifier import StanceClassifier
from src.scraping.pboc_scraper import fetch_article_text

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "title", "url", "published", "scraped_at", "text_length",
    "segment_type", "stance", "confidence", "reasoning",
]

LISTINGS = Path("data/processed/corpus_2022_2024_listings.csv")


def load_done(out: Path) -> set[str]:
    """Return set of URLs already successfully classified (segment_type not null)."""
    if not out.exists():
        return set()
    df = pd.read_csv(out, usecols=["url", "segment_type"])
    done = df[df["segment_type"].notna() & (df["segment_type"] != "")]
    return set(done["url"].dropna())


def run(out: Path, listings_path: Path = LISTINGS) -> None:
    listings = pd.read_csv(listings_path)
    done = load_done(out)
    todo = listings[~listings["url"].isin(done)]

    logger.info("Total: %d | Done: %d | Remaining: %d", len(listings), len(done), len(todo))

    if todo.empty:
        logger.info("Nothing to do.")
        return

    clf = StanceClassifier()
    write_header = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, (_, row) in enumerate(todo.iterrows()):
            if row.get("text_length", -1) == 0:
                seg = stance = reasoning = ""
                conf = 0.0
                text_len = 0
            else:
                try:
                    text = fetch_article_text(row["url"])
                except Exception as e:
                    logger.warning("[%d] fetch failed: %s", i + 1, e)
                    text = ""

                text_len = len(text)
                if text:
                    try:
                        r = clf.classify(text)
                        seg, stance, conf, reasoning = (
                            r.segment_type, r.stance, r.confidence, r.reasoning
                        )
                    except Exception as e:
                        logger.warning("[%d] classify failed: %s", i + 1, e)
                        seg = stance = reasoning = ""
                        conf = 0.0
                else:
                    seg = stance = reasoning = ""
                    conf = 0.0

            writer.writerow({
                "title":        row["title"],
                "url":          row["url"],
                "published":    row["published"],
                "scraped_at":   row.get("scraped_at", ""),
                "text_length":  text_len,
                "segment_type": seg,
                "stance":       stance,
                "confidence":   conf,
                "reasoning":    reasoning,
            })
            f.flush()

            if (i + 1) % 50 == 0:
                logger.info("[%d / %d] %s | %s %s", i + 1, len(todo),
                            str(row["published"])[:10], seg, stance)
            time.sleep(1)

    logger.info("Done. Output: %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("data/processed/corpus_classified.csv"))
    parser.add_argument("--listings", type=Path, default=LISTINGS)
    args = parser.parse_args()
    run(args.out, listings_path=args.listings)
