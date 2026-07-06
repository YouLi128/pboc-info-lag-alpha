"""
Pilot data collection pipeline.

Scrapes PBOC listing pages, fetches full article text, runs LLM stance
classification, and saves results to data/processed/pilot_classified.csv.

Usage:
    python -m src.scraping.pilot_collect --pages 5 --out data/processed/pilot_classified.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.llm_processing.stance_classifier import StanceClassifier
from src.scraping.pboc_scraper import fetch_article_text, scrape_listings

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "title", "url", "published", "scraped_at",
    "text_length", "segment_type", "stance", "confidence", "reasoning",
]


def run(pages: int, out: Path, skip_llm: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    clf = None if skip_llm else StanceClassifier()

    total = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for article in scrape_listings(max_pages=pages):
            logger.info("[%d] %s  (%s)", total + 1, article["title"][:40], article["published"])

            # Full text
            try:
                text = fetch_article_text(article["url"])
            except Exception as exc:
                logger.warning("Full-text fetch failed: %s", exc)
                text = ""

            # LLM classification
            if clf and text:
                try:
                    result = clf.classify(text)
                    seg, stance, conf, reasoning = (
                        result.segment_type, result.stance,
                        result.confidence, result.reasoning,
                    )
                except Exception as exc:
                    logger.warning("Classification failed: %s", exc)
                    seg = stance = reasoning = ""
                    conf = 0.0
            else:
                seg = stance = reasoning = ""
                conf = 0.0

            writer.writerow({
                **article,
                "text_length": len(text),
                "segment_type": seg,
                "stance": stance,
                "confidence": conf,
                "reasoning": reasoning,
            })
            f.flush()
            total += 1
            time.sleep(1)  # polite gap between articles

    logger.info("Done. %d articles saved to %s", total, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=3, help="Number of listing pages to scrape")
    parser.add_argument("--out", type=Path, default=Path("data/processed/pilot_classified.csv"))
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM classification (scrape only)")
    args = parser.parse_args()
    run(pages=args.pages, out=args.out, skip_llm=args.skip_llm)
