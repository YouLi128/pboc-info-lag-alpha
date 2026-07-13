"""
Dictionary-baseline event study — parallels build_events() from
event_study_pipeline.py but sources the stance signal from keyword
counts (dictionary_baseline.py) instead of the LLM classifier. Same
60-day rolling-baseline surprise construction, same downstream
alignment/Granger/placebo machinery, so results are directly comparable
to the LLM-based branch.

Usage:
    python -m src.analysis.dictionary_event_study
"""

from __future__ import annotations

import logging

import pandas as pd

from src.analysis.event_study_pipeline import extract_release_time, CST
from src.signal_construction.dictionary_baseline import score_corpus

logger = logging.getLogger(__name__)


def build_events_dict(fulltext_csv: str, baseline_window_days: int = 60) -> pd.DataFrame:
    df = score_corpus(fulltext_csv)

    def _get_timestamp(row):
        url_ts = extract_release_time(row["url"])
        pub_date = pd.to_datetime(row["published"]).date()
        if url_ts is not None and url_ts.date() == pub_date:
            return url_ts
        d = pd.to_datetime(row["published"])
        return pd.Timestamp(d.year, d.month, d.day, 10, 0, 0, tzinfo=CST).tz_convert("UTC")

    df["release_utc"] = pd.to_datetime(df.apply(_get_timestamp, axis=1), utc=True)
    df["event_date"] = df["release_utc"].dt.date.astype(str)

    agg = df.groupby("event_date").agg(
        release_utc  = ("release_utc", "min"),
        n_articles   = ("url", "count"),
        stance_score = ("dict_score", "mean"),
    ).reset_index()

    agg = agg.sort_values("release_utc").reset_index(drop=True)
    ts_indexed = agg.set_index("release_utc")["stance_score"]
    baseline = ts_indexed.rolling(f"{baseline_window_days}D", closed="left", min_periods=1).mean()
    agg["baseline"] = baseline.fillna(0).to_numpy()
    agg["surprise"] = agg["stance_score"] - agg["baseline"]

    logger.info("Dictionary events after dedup: %d (from %d articles)", len(agg), len(df))
    return agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.analysis.event_study_pipeline import align_hourly, run_granger_daily
    from src.analysis.repo_rate_event_study import align_daily_rate
    from src.analysis.cgb_yield_event_study import align_daily_cgb
    from src.analysis.ndrc_event_study import align_daily_csi300
    from src.analysis.volatility_channel_test import volatility_regression

    events = build_events_dict("data/processed/pboc_fg_fulltext.csv")
    print(f"\n=== Dictionary baseline: {len(events)} events ===")

    print("\n--- CNH hourly ---")
    aligned = align_hourly(events, "data/raw/cnh_1h_all_events.csv")
    for w in [1, 2, 4, 8, 24]:
        r = run_granger_daily(aligned, y_col=f"ret_{w}h", x_col="surprise")
        vr = volatility_regression(aligned, f"ret_{w}h")
        print(f"  ret_{w}h: dir min_p={r.get('min_pval', 1):.4f} | vol p={vr.get('ols_pval', 1):.4f}")

    print("\n--- FDR007 ---")
    aligned = align_daily_rate(events, "data/raw/repo_rate_daily.csv")
    for w in [0, 1, 2]:
        r = run_granger_daily(aligned, y_col=f"chg_{w}d_bps", x_col="surprise")
        vr = volatility_regression(aligned, f"chg_{w}d_bps")
        print(f"  chg_{w}d: dir min_p={r.get('min_pval', 1):.4f} | vol p={vr.get('ols_pval', 1):.4f}")

    print("\n--- CGB 10y ---")
    aligned = align_daily_cgb(events, "data/raw/cgb_yield_daily.csv")
    for w in [0, 1, 2]:
        r = run_granger_daily(aligned, y_col=f"chg_{w}d_bps", x_col="surprise")
        vr = volatility_regression(aligned, f"chg_{w}d_bps")
        print(f"  chg_{w}d: dir min_p={r.get('min_pval', 1):.4f} | vol p={vr.get('ols_pval', 1):.4f}")

    print("\n--- CSI 300 ---")
    aligned = align_daily_csi300(events, "data/raw/csi300_daily.csv")
    for w in [0, 1, 2]:
        r = run_granger_daily(aligned, y_col=f"ret_{w}d", x_col="surprise")
        vr = volatility_regression(aligned, f"ret_{w}d")
        print(f"  ret_{w}d: dir min_p={r.get('min_pval', 1):.4f} | vol p={vr.get('ols_pval', 1):.4f}")

    events.to_csv("data/processed/pboc_dict_events.csv", index=False)
