"""
CNH-CNY basis event study — capital-control/expectation-gap channel.

The onshore-offshore spread (CNH - CNY fixing) is a cleaner proxy for
capital-control and devaluation-expectation shifts than either rate
alone: it should widen when offshore markets price in more depreciation
risk than the official fixing reflects. Built from cnh_1h_all_events.csv
(offshore, event-window coverage only) joined against
usdcny_fixing_daily.csv (onshore PBOC fixing, continuous 2019-2026).

Basis coverage is therefore capped at whatever days the CNH hourly
event-window download actually reached (see events_aligned_hourly_full.csv
for the same 173/205 coverage figure).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.event_study_pipeline import build_events, run_granger_daily

logger = logging.getLogger(__name__)


def build_basis_series(cnh_1h_csv: str, usdcny_csv: str) -> pd.DataFrame:
    cnh = pd.read_csv(cnh_1h_csv, index_col=0, parse_dates=True)
    if cnh.index.tz is not None:
        cnh.index = cnh.index.tz_convert(None)
    daily_cnh = cnh["close"].resample("1D").last().dropna()

    cny = pd.read_csv(usdcny_csv, index_col=0, parse_dates=True)

    merged = pd.DataFrame({"cnh": daily_cnh}).join(cny, how="inner")
    merged["basis_pips"] = (merged["cnh"] - merged["usdcny_fixing"]) * 10000
    return merged


def align_daily_basis(events: pd.DataFrame, basis: pd.DataFrame,
                      windows: list[int] = [0, 1, 2]) -> pd.DataFrame:
    mkt = basis.copy()
    mkt.index = pd.to_datetime(mkt.index, utc=True)
    mkt["chg"] = mkt["basis_pips"].diff()

    mkt_start, mkt_end = mkt.index[0], mkt.index[-1]

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["release_utc"]
        row = ev.to_dict()
        if t0 < mkt_start or t0 > mkt_end:
            for w in windows:
                row[f"chg_{w}d_pips"] = np.nan
            rows.append(row)
            continue
        future = mkt[mkt.index >= t0]
        for w in windows:
            row[f"chg_{w}d_pips"] = future["chg"].iloc[w] if len(future) > w else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    basis = build_basis_series("data/raw/cnh_1h_all_events.csv", "data/raw/usdcny_fixing_daily.csv")
    logger.info("Basis series: %d days (%s ~ %s)", len(basis), basis.index.min(), basis.index.max())

    events = build_events("data/processed/corpus_classified.csv")
    print(f"\n=== {len(events)} deduplicated PBOC forward_guidance events ===")

    aligned = align_daily_basis(events, basis)
    n_valid = aligned["chg_0d_pips"].notna().sum()
    print(f"{n_valid} / {len(aligned)} events fall inside the basis data window")

    for w in [0, 1, 2]:
        col = f"chg_{w}d_pips"
        print(f"\n=== Granger causality: surprise -> {col} (CNH-CNY basis) ===")
        print(run_granger_daily(aligned, y_col=col, x_col="surprise"))

    out_path = "data/processed/events_aligned_cnh_cny_basis.csv"
    aligned.to_csv(out_path, index=False)
    print(f"\nSaved aligned event table to {out_path}")
