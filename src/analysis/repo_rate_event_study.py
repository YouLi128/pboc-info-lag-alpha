"""
Repo-rate (FDR007) event study — cross-validation branch.

Same forward_guidance events as the CNH study, aligned instead to the
domestic interbank repo fixing rate (FDR007). Liquidity-related PBOC
guidance should transmit faster/cleaner into short-term money-market
rates than into offshore FX, so a significant result here is evidence
that the LLM stance signal carries real information rather than noise.

Rates are level series (%), so the outcome is a basis-point change
(diff), not a pct_change like FX/equity returns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.event_study_pipeline import build_events, run_granger_daily

logger = logging.getLogger(__name__)

RATE_COL = "FDR007"


def align_daily_rate(events: pd.DataFrame, repo_rate_csv: str,
                     rate_col: str = RATE_COL,
                     windows: list[int] = [0, 1, 2]) -> pd.DataFrame:
    """
    Compute forward changes (in bps) in the repo rate over windows
    (in trading days) following each event.
    """
    mkt = pd.read_csv(repo_rate_csv, parse_dates=["date"]).set_index("date").sort_index()
    mkt.index = pd.to_datetime(mkt.index, utc=True)
    mkt["chg"] = mkt[rate_col].diff() * 100  # percentage points -> bps

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

    aligned = align_daily_rate(events, "data/raw/repo_rate_daily.csv")
    n_valid = aligned["chg_0d_bps"].notna().sum()
    print(f"\n{n_valid} / {len(aligned)} events fall inside the FDR007 data window (2019-01 ~ present)")

    for w in [0, 1, 2]:
        col = f"chg_{w}d_bps"
        print(f"\n=== Granger causality: surprise -> {col} ===")
        print(run_granger_daily(aligned, y_col=col, x_col="surprise"))

    out_path = "data/processed/events_aligned_repo_rate.csv"
    aligned.to_csv(out_path, index=False)
    print(f"\nSaved aligned event table to {out_path}")
