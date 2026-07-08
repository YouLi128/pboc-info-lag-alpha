"""
Interbank repo rate loader — China domestic rates market.

Extension to the CNH-only event study: PBOC forward guidance about
liquidity/reserve requirements should transmit almost immediately into
short-term money market rates, giving a cleaner (less noisy) validation
channel than offshore FX.

Source: chinamoney.com.cn via the akshare wrapper (`repo_rate_hist`).
No API key required. Server caps each query to roughly one calendar year,
so fetch_repo_rate_range() chunks by year and concatenates.

Columns returned (all in %, daily):
    FR001, FR007, FR014     — pledged repo fixing rate (all institutions)
    FDR001, FDR007, FDR014  — pledged repo fixing rate, deposit-type
                              institutions only (FDR007 is the DR007 proxy
                              PBOC itself watches most closely)
"""

from __future__ import annotations

import logging
from datetime import date

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

RATE_COLUMNS = ["FR001", "FR007", "FR014", "FDR001", "FDR007", "FDR014"]


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    y = start.year
    while y <= end.year:
        chunk_start = max(start, date(y, 1, 1))
        chunk_end = min(end, date(y, 12, 31))
        chunks.append((chunk_start, chunk_end))
        y += 1
    return chunks


def fetch_repo_rate_range(start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily FR/FDR repo fixing rates for [start, end], chunked by year.

    Returns:
        DataFrame indexed by date (tz-naive, China local trading day) with
        columns RATE_COLUMNS (float, %).
    """
    parts = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        s = chunk_start.strftime("%Y%m%d")
        e = chunk_end.strftime("%Y%m%d")
        logger.info("Fetching repo rates %s → %s", s, e)
        df = ak.repo_rate_hist(start_date=s, end_date=e)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset="date").set_index("date").sort_index()
    logger.info("Fetched %d daily repo-rate rows (%s ~ %s)", len(out), out.index.min(), out.index.max())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch_repo_rate_range(date(2019, 1, 1), date.today())
    out_path = "data/raw/repo_rate_daily.csv"
    df.to_csv(out_path)
    print(df.tail())
    print(f"\nSaved {len(df)} rows to {out_path}")
