"""
Generic continuous (-10..+10) LLM stance scorer runner — parallels
classify_pboc_continuous.py but works for any source with a cached
fulltext CSV (ndrc_fg_fulltext.csv, mof_fg_fulltext.csv) and a matching
prompt variant in continuous_scorer.PROMPTS.

Usage:
    python -m src.scraping.classify_continuous --source ndrc
    python -m src.scraping.classify_continuous --source mof
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
from src.llm_processing.continuous_scorer import ContinuousScorer

logger = logging.getLogger(__name__)

FIELDNAMES = ["url", "published", "score", "confidence", "reasoning"]


def load_done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    df = pd.read_csv(out, usecols=["url", "confidence", "reasoning"])
    done = df[df["confidence"].notna() & (df["reasoning"] != "error")]
    return set(done["url"].dropna())


def run(source: str, fulltext_csv: str, out: Path) -> None:
    listings = pd.read_csv(fulltext_csv)
    done = load_done(out)
    todo = listings[~listings["url"].isin(done)]
    logger.info("[%s] Total: %d | Done: %d | Remaining: %d", source, len(listings), len(done), len(todo))

    if todo.empty:
        logger.info("Nothing to do.")
        return

    scorer = ContinuousScorer(source=source)
    write_header = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("a", newline="", encoding="utf-8") as f:
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
                logger.info("[%s] [%d / %d] score=%.1f", source, i + 1, len(todo), score)
            time.sleep(0.5)

    logger.info("[%s] Done. Output: %s", source, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ndrc", "mof"], required=True)
    args = parser.parse_args()
    fulltext_csv = f"data/processed/{args.source}_fg_fulltext.csv"
    out = Path(f"data/processed/{args.source}_fg_continuous_scored.csv")
    run(args.source, fulltext_csv, out)
