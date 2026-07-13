"""
Score all 248 PBOC forward_guidance articles with the continuous
-10..+10 LLM stance scorer. Reuses cached text from pboc_fg_fulltext.csv
(built for the dictionary baseline) — no re-scraping needed.

Usage:
    python -m src.scraping.classify_pboc_continuous
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from src.llm_processing.continuous_scorer import ContinuousScorer

logger = logging.getLogger(__name__)

FULLTEXT_CSV = "data/processed/pboc_fg_fulltext.csv"
OUT = Path("data/processed/pboc_fg_continuous_scored.csv")
FIELDNAMES = ["url", "published", "score", "confidence", "reasoning"]


def load_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    df = pd.read_csv(out, usecols=["url", "confidence"])
    done = df[df["confidence"].notna()]
    return set(done["url"].dropna())


def run() -> None:
    listings = pd.read_csv(FULLTEXT_CSV)
    done = load_done(OUT)
    todo = listings[~listings["url"].isin(done)]
    logger.info("Total: %d | Done: %d | Remaining: %d", len(listings), len(done), len(todo))

    if todo.empty:
        logger.info("Nothing to do.")
        return

    scorer = ContinuousScorer()
    write_header = not OUT.exists()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for i, (_, row) in enumerate(todo.iterrows()):
            text = row.get("text", "")
            if not isinstance(text, str) or not text:
                score, conf, reasoning = 0.0, 0.0, "empty_text"
            else:
                try:
                    r = scorer.score(text)
                    score, conf, reasoning = r.score, r.confidence, r.reasoning
                except Exception as e:
                    logger.warning("[%d] scoring failed: %s", i + 1, e)
                    score, conf, reasoning = 0.0, 0.0, "error"

            writer.writerow({
                "url": row["url"], "published": row["published"],
                "score": score, "confidence": conf, "reasoning": reasoning,
            })
            f.flush()

            if (i + 1) % 25 == 0:
                logger.info("[%d / %d] score=%.1f", i + 1, len(todo), score)
            time.sleep(0.5)

    logger.info("Done. Output: %s", OUT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
