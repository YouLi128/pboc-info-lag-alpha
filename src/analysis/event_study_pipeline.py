"""
End-to-end event study pipeline.

Steps:
  1. Extract precise publication timestamps from PBOC article URLs
  2. Deduplicate same-day events by aggregating surprise scores
  3. Align to hourly CNH data for short-horizon return windows
  4. Run Granger causality (daily) and event study (hourly)
"""

from __future__ import annotations

import re
import logging
from datetime import timezone

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)

CST = timezone(pd.Timedelta(hours=8))
STANCE_SCORE = {"dovish": -1.0, "hawkish": 1.0, "neutral": 0.0}


# ---------------------------------------------------------------------------
# Step 1: extract timestamps from URLs
# ---------------------------------------------------------------------------

def extract_release_time(url: str) -> pd.Timestamp | None:
    """
    PBOC article URLs embed the publish time as YYYYMMDDHHMMSS (Beijing time).
    Example: .../2024092418330271304/... → 2024-09-24 18:33:02 CST → UTC
    """
    m = re.search(r"/(\d{14})\d*/index", url)
    if not m:
        return None
    s = m.group(1)
    try:
        ts_cst = pd.Timestamp(
            int(s[:4]), int(s[4:6]), int(s[6:8]),
            int(s[8:10]), int(s[10:12]), int(s[12:14]),
            tzinfo=CST,
        )
        return ts_cst.tz_convert("UTC")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 2: build and deduplicate event table
# ---------------------------------------------------------------------------

def build_events(classified_csv: str) -> pd.DataFrame:
    """
    Load classified articles, extract precise timestamps, compute surprise,
    and aggregate multiple same-day articles into one event per day.
    """
    df = pd.read_csv(classified_csv)
    df = df[df["segment_type"] == "forward_guidance"].copy()

    # Extract timestamp from URL only when URL date matches published date
    # (historical articles were migrated in 2025, so URL timestamp = migration date, not publish date)
    def _get_timestamp(row):
        url_ts = extract_release_time(row["url"])
        pub_date = pd.to_datetime(row["published"]).date()
        if url_ts is not None and url_ts.date() == pub_date:
            return url_ts  # URL timestamp is reliable
        # Fallback: published date at 10:00 CST (typical PBOC release window)
        d = pd.to_datetime(row["published"])
        return pd.Timestamp(d.year, d.month, d.day, 10, 0, 0, tzinfo=CST).tz_convert("UTC")

    df["release_utc"] = pd.to_datetime(
        df.apply(_get_timestamp, axis=1), utc=True
    )

    df["stance_score"] = df["stance"].map(STANCE_SCORE).fillna(0.0)
    df["weighted_score"] = df["stance_score"] * df["confidence"]

    # Aggregate by calendar date (one event per day)
    df["event_date"] = df["release_utc"].dt.date.astype(str)
    agg = df.groupby("event_date").agg(
        release_utc   = ("release_utc", "min"),   # earliest article that day
        n_articles    = ("url", "count"),
        stance_score  = ("weighted_score", "mean"),
        confidence    = ("confidence", "mean"),
        stances       = ("stance", lambda x: "/".join(sorted(set(x)))),
    ).reset_index()

    # Rolling baseline surprise (window = 5 prior events)
    agg = agg.sort_values("release_utc").reset_index(drop=True)
    agg["baseline"] = agg["stance_score"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    agg["surprise"] = agg["stance_score"] - agg["baseline"]

    logger.info("Events after dedup: %d (from %d articles)", len(agg), len(df))
    return agg


# ---------------------------------------------------------------------------
# Step 3: align to hourly CNH market data
# ---------------------------------------------------------------------------

def align_hourly(events: pd.DataFrame, cnh_1h_csv: str,
                 windows: list[int] = [1, 2, 4, 8, 24]) -> pd.DataFrame:
    """
    Compute forward returns over windows (in hours) after each event.
    Uses hourly CNH data from Dukascopy.
    """
    mkt = pd.read_csv(cnh_1h_csv, index_col=0, parse_dates=True)
    if mkt.index.tz is None:
        mkt.index = mkt.index.tz_localize("UTC")

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["release_utc"]
        row = ev.to_dict()

        future = mkt[mkt.index >= t0]
        if future.empty:
            for w in windows:
                row[f"ret_{w}h"] = np.nan
            rows.append(row)
            continue

        p0 = future.iloc[0]["close"]
        for w in windows:
            t_end = t0 + pd.Timedelta(hours=w)
            sl = mkt[(mkt.index >= t0) & (mkt.index <= t_end)]
            row[f"ret_{w}h"] = (sl.iloc[-1]["close"] - p0) / p0 if not sl.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def align_daily(events: pd.DataFrame, cnh_1d_csv: str,
                windows: list[int] = [0, 1, 2]) -> pd.DataFrame:
    """Compute forward returns over windows (in trading days)."""
    mkt = pd.read_csv(cnh_1d_csv, parse_dates=["date"]).set_index("date").sort_index()
    mkt.index = pd.to_datetime(mkt.index, utc=True)
    mkt["ret"] = mkt["close"].pct_change()

    rows = []
    for _, ev in events.iterrows():
        t0 = ev["release_utc"]
        row = ev.to_dict()
        future = mkt[mkt.index >= t0]
        for w in windows:
            row[f"ret_{w}d"] = future["ret"].iloc[w] if len(future) > w else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 4: Granger causality (daily)
# ---------------------------------------------------------------------------

def run_granger_daily(aligned: pd.DataFrame,
                      y_col: str = "ret_1d",
                      x_col: str = "surprise",
                      max_lag: int = 3) -> dict:
    data = aligned[[y_col, x_col]].dropna()
    if len(data) < max_lag * 4:
        return {"error": "insufficient_data", "n_obs": len(data)}

    res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    out = {f"lag_{l}_pval": res[l][0]["ssr_ftest"][1] for l in res}
    out["min_pval"] = min(v for v in out.values())
    out["significant_5pct"] = out["min_pval"] < 0.05
    out["n_obs"] = len(data)
    return out


# ---------------------------------------------------------------------------
# Step 5: event study summary
# ---------------------------------------------------------------------------

def event_study_summary(aligned: pd.DataFrame,
                        ret_cols: list[str]) -> pd.DataFrame:
    """Mean returns by surprise direction."""
    df = aligned.copy()
    df["direction"] = pd.cut(
        df["surprise"],
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=["dovish_surprise", "neutral", "hawkish_surprise"],
    )
    return df.groupby("direction", observed=True)[ret_cols].agg(["mean", "count"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = build_events("data/processed/policy_window_classified.csv")
    print(f"\n=== {len(events)} deduplicated events ===")
    print(events[["event_date", "n_articles", "stances", "surprise"]].to_string())

    # Hourly event study
    aligned_h = align_hourly(events, "data/raw/cnh_1h_policy_windows.csv")
    ret_cols_h = [c for c in aligned_h.columns if c.startswith("ret_") and c.endswith("h")]
    print(f"\n=== Event Study (hourly windows) ===")
    print(event_study_summary(aligned_h, ret_cols_h).to_string())

    # Daily Granger
    aligned_d = align_daily(events, "data/raw/cnh_1d_2022_2024.csv")
    print(f"\n=== Granger Causality: surprise → next-day CNH return ===")
    print(run_granger_daily(aligned_d, y_col="ret_1d", x_col="surprise"))

    # Save
    aligned_h.to_csv("data/processed/events_aligned_hourly.csv", index=False)
    aligned_d.to_csv("data/processed/events_aligned_daily.csv", index=False)
    print("\nSaved aligned event tables to data/processed/")
