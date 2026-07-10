"""
China Government Bond (CGB) yield curve loader — long-end rate channel.

Complements FDR007 (short-end repo rate): PBOC liquidity guidance should
transmit fastest to short rates, but forward guidance about the broader
policy stance (growth outlook, structural reform) may show up more in the
10-year yield, which prices in expectations over a much longer horizon.

Source: ChinaBond (中债登) via the akshare wrapper (`bond_china_yield`).
No API key required. Server caps each query to <1 year, so
fetch_cgb_yield_range() chunks by year and concatenates.
"""

from __future__ import annotations

import logging
from datetime import date

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

CURVE_NAME = "中债国债收益率曲线"
TENOR_COLS = ["3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"]


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    y = start.year
    while y <= end.year:
        chunk_start = max(start, date(y, 1, 1))
        chunk_end = min(end, date(y, 12, 31))
        chunks.append((chunk_start, chunk_end))
        y += 1
    return chunks


def fetch_cgb_yield_range(start: date, end: date) -> pd.DataFrame:
    """
    Fetch the daily CGB (government bond) yield curve for [start, end],
    chunked by year, filtered to the 中债国债收益率曲线 series.

    Returns:
        DataFrame indexed by date with columns TENOR_COLS (float, %).
    """
    parts = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        s = chunk_start.strftime("%Y%m%d")
        e = chunk_end.strftime("%Y%m%d")
        logger.info("Fetching CGB yield curve %s -> %s", s, e)
        df = ak.bond_china_yield(start_date=s, end_date=e)
        df = df[df["曲线名称"] == CURVE_NAME]
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["日期"])
    out = out.drop_duplicates(subset="date").set_index("date").sort_index()
    out = out[TENOR_COLS]
    out.columns = ["3m", "6m", "1y", "3y", "5y", "7y", "10y", "30y"]
    logger.info("Fetched %d daily CGB yield rows (%s ~ %s)", len(out), out.index.min(), out.index.max())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch_cgb_yield_range(date(2019, 1, 1), date.today())
    out_path = "data/raw/cgb_yield_daily.csv"
    df.to_csv(out_path)
    print(df[["10y"]].tail())
    print(f"\nSaved {len(df)} rows to {out_path}")
