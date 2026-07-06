"""
Granger causality and event-study tests.

Tests H1: surprise score Granger-causes CNH returns.
Tests H2: channel heterogeneity (headline vs. non-headline).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Granger causality (H1)
# ---------------------------------------------------------------------------


def run_granger(
    df: pd.DataFrame,
    y_col: str = "ret_15m",
    x_col: str = "surprise",
    max_lag: int = 5,
    significance: float = 0.05,
) -> dict:
    """
    Test whether x_col Granger-causes y_col.

    Args:
        df:           Event-aligned DataFrame with both columns.
        y_col:        Dependent variable (market return).
        x_col:        Candidate cause (surprise score).
        max_lag:      Maximum lag order to test.
        significance: p-value threshold.

    Returns:
        Dict with per-lag p-values and an overall 'significant' flag.
    """
    data = df[[y_col, x_col]].dropna()
    if len(data) < max_lag * 3:
        logger.warning(
            "Too few observations (%d) for Granger test with max_lag=%d",
            len(data),
            max_lag,
        )
        return {"error": "insufficient_data", "n_obs": len(data)}

    results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    summary = {}
    for lag, res in results.items():
        # res[0] contains test stats; use F-test p-value
        pval = res[0]["ssr_ftest"][1]
        summary[f"lag_{lag}_pval"] = pval

    min_pval = min(v for k, v in summary.items() if k.endswith("_pval"))
    summary["min_pval"] = min_pval
    summary["significant"] = min_pval < significance
    summary["n_obs"] = len(data)

    return summary


# ---------------------------------------------------------------------------
# Event study (H1, H2)
# ---------------------------------------------------------------------------


def event_study(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute average abnormal returns around PBOC events.

    Args:
        df:        Event-aligned DataFrame with ret_{w}m columns and 'surprise'.
        windows:   Forward-return windows in minutes (must match column names).
        group_col: Optional column to split results by (e.g. 'channel' for H2).

    Returns:
        DataFrame of mean returns for dovish / hawkish / neutral surprises,
        optionally stratified by group_col.
    """
    if windows is None:
        windows = [5, 15, 30, 60]

    ret_cols = [f"ret_{w}m" for w in windows]
    df = df.copy()

    # Bin by surprise direction
    df["surprise_direction"] = pd.cut(
        df["surprise"],
        bins=[-np.inf, -1e-6, 1e-6, np.inf],
        labels=["dovish_surprise", "neutral", "hawkish_surprise"],
    )

    groupby_cols = ["surprise_direction"]
    if group_col and group_col in df.columns:
        groupby_cols = [group_col] + groupby_cols

    summary = (
        df.groupby(groupby_cols, observed=True)[ret_cols]
        .agg(["mean", "std", "count"])
    )
    return summary


# ---------------------------------------------------------------------------
# Lag-shrinkage test (H3 stretch)
# ---------------------------------------------------------------------------


def test_lag_shrinkage(
    df: pd.DataFrame,
    date_col: str = "release_time",
    significance_col: str = "min_pval",
    n_splits: int = 3,
) -> pd.DataFrame:
    """
    Split the sample into equal time periods and compare mean surprise-return
    correlations across periods.  A declining pattern would support H3.

    TODO: Replace with a formal Chow test or rolling-window regression once
    sample size is large enough.
    """
    df = df.copy()
    df["period"] = pd.qcut(df[date_col].astype(np.int64), q=n_splits, labels=False)

    # TODO: compute period-level Granger p-values or R² and compare.
    # Placeholder: return period-mean surprise scores.
    return df.groupby("period")["surprise"].describe()
