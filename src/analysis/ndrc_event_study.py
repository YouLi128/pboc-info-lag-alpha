"""
NDRC -> CSI 300 event study — broad-market extension branch.

Separate research question from the core CNH cross-border-lag study
(see repo README): does a wider policy-document corpus predict the
broad domestic index. Reuses build_events()/run_granger_daily() from
event_study_pipeline.py; only the market series changes (CSI 300 pct
returns instead of CNH).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.event_study_pipeline import build_events, run_granger_daily

logger = logging.getLogger(__name__)


def align_daily_csi300(events: pd.DataFrame, csi300_csv: str,
                       windows: list[int] = [0, 1, 2]) -> pd.DataFrame:
    mkt = pd.read_csv(csi300_csv, index_col=0, parse_dates=True)
    mkt.index = pd.to_datetime(mkt.index, utc=True)
    mkt["ret"] = mkt["close"].pct_change()

    mkt_start, mkt_end = mkt.index[0], mkt.index[-1]

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["release_utc"]
        row = ev.to_dict()
        if t0 < mkt_start or t0 > mkt_end:
            for w in windows:
                row[f"ret_{w}d"] = np.nan
            rows.append(row)
            continue
        future = mkt[mkt.index >= t0]
        for w in windows:
            row[f"ret_{w}d"] = future["ret"].iloc[w] if len(future) > w else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = build_events("data/processed/ndrc_classified.csv")
    print(f"\n=== {len(events)} deduplicated NDRC forward_guidance events ===")

    aligned = align_daily_csi300(events, "data/raw/csi300_daily.csv")
    n_valid = aligned["ret_0d"].notna().sum()
    print(f"{n_valid} / {len(aligned)} events fall inside the CSI 300 data window")

    for w in [0, 1, 2]:
        col = f"ret_{w}d"
        print(f"\n=== Granger causality: surprise -> {col} ===")
        print(run_granger_daily(aligned, y_col=col, x_col="surprise"))

    out_path = "data/processed/events_aligned_csi300.csv"
    aligned.to_csv(out_path, index=False)
    print(f"\nSaved aligned event table to {out_path}")
