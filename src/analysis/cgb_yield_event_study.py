"""
CGB 10-year yield event study — long-end rate channel.

Same forward_guidance events as the CNH/repo-rate studies, aligned to the
10-year China government bond yield instead. FDR007 (repo_rate_event_study.py)
tests the short-end liquidity channel; this tests whether the same
signal moves longer-horizon growth/policy-stance expectations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.event_study_pipeline import build_events, run_granger_daily

logger = logging.getLogger(__name__)

TENOR = "10y"


def align_daily_cgb(events: pd.DataFrame, cgb_csv: str,
                    tenor: str = TENOR,
                    windows: list[int] = [0, 1, 2]) -> pd.DataFrame:
    """Compute forward changes (in bps) in the CGB yield following each event."""
    mkt = pd.read_csv(cgb_csv, parse_dates=["date"]).set_index("date").sort_index()
    mkt.index = pd.to_datetime(mkt.index, utc=True)
    mkt["chg"] = mkt[tenor].diff() * 100  # percentage points -> bps

    mkt_start, mkt_end = mkt.index[0], mkt.index[-1]

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["release_utc"]
        row = ev.to_dict()
        if t0 < mkt_start or t0 > mkt_end:
            for w in windows:
                row[f"chg_{w}d_bps"] = np.nan
            rows.append(row)
            continue
        future = mkt[mkt.index >= t0]
        for w in windows:
            row[f"chg_{w}d_bps"] = future["chg"].iloc[w] if len(future) > w else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = build_events("data/processed/corpus_classified.csv")
    print(f"\n=== {len(events)} deduplicated forward_guidance events ===")

    aligned = align_daily_cgb(events, "data/raw/cgb_yield_daily.csv")
    n_valid = aligned["chg_0d_bps"].notna().sum()
    print(f"\n{n_valid} / {len(aligned)} events fall inside the CGB yield data window")

    for w in [0, 1, 2]:
        col = f"chg_{w}d_bps"
        print(f"\n=== Granger causality: surprise -> {col} (10y CGB yield) ===")
        print(run_granger_daily(aligned, y_col=col, x_col="surprise"))

    out_path = "data/processed/events_aligned_cgb_yield.csv"
    aligned.to_csv(out_path, index=False)
    print(f"\nSaved aligned event table to {out_path}")
