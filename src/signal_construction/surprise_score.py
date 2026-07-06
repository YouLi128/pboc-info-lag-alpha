"""
Signal construction: PBOC communication surprise scores.

Combines LLM stance labels with timestamps to produce a per-event
"surprise" signal, then aligns it to market data for downstream analysis.

Surprise is defined as the deviation of the current stance from a rolling
expectation baseline — a simple approach for v0, following Gürkaynak et al.
(2005) in spirit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Stance → numeric encoding
STANCE_SCORE = {"dovish": -1.0, "hawkish": 1.0, "neutral": 0.0}


# ---------------------------------------------------------------------------
# Surprise score construction
# ---------------------------------------------------------------------------


def encode_stance(stance: str) -> float:
    """Map a stance label to a signed numeric score."""
    return STANCE_SCORE.get(stance, 0.0)


def build_event_table(classifications: list[dict]) -> pd.DataFrame:
    """
    Convert a list of classification result dicts to an event DataFrame.

    Expected input keys per dict:
        release_time (str | datetime, UTC): timestamp of PBOC document release
        segment_type (str)
        stance (str)
        confidence (float)

    Returns DataFrame with columns:
        release_time, stance_score, confidence, is_forward_guidance
    """
    df = pd.DataFrame(classifications)
    df["release_time"] = pd.to_datetime(df["release_time"], utc=True)
    df["stance_score"] = df["stance"].map(STANCE_SCORE).fillna(0.0)
    df["is_forward_guidance"] = df["segment_type"] == "forward_guidance"

    # Weight score by confidence and forward-guidance flag
    df["weighted_score"] = (
        df["stance_score"] * df["confidence"] * df["is_forward_guidance"].astype(float)
    )

    df = df.sort_values("release_time").reset_index(drop=True)
    return df


def compute_surprise(
    events: pd.DataFrame,
    baseline_window: int = 5,
) -> pd.DataFrame:
    """
    Compute a rolling-mean surprise score.

    Surprise_t = weighted_score_t − mean(weighted_score_{t-baseline_window : t-1})

    Args:
        events:          Output of build_event_table().
        baseline_window: Number of prior events to use as expectation baseline.

    Returns:
        events with an added "surprise" column.
    """
    events = events.copy()
    rolling_mean = (
        events["weighted_score"]
        .shift(1)
        .rolling(window=baseline_window, min_periods=1)
        .mean()
    )
    events["baseline"] = rolling_mean
    events["surprise"] = events["weighted_score"] - rolling_mean.fillna(0.0)
    return events


# ---------------------------------------------------------------------------
# Timestamp lag alignment
# ---------------------------------------------------------------------------


def align_to_market(
    events: pd.DataFrame,
    market: pd.DataFrame,
    windows_minutes: list[int] | None = None,
) -> pd.DataFrame:
    """
    For each event, compute forward returns over specified windows.

    Args:
        events:          DataFrame with 'release_time' column (UTC).
        market:          OHLCV DataFrame indexed by UTC DatetimeIndex.
        windows_minutes: List of forward-return windows in minutes.

    Returns:
        events with added columns: ret_{w}m for each window w.
    """
    if windows_minutes is None:
        windows_minutes = [5, 15, 30, 60]

    results = []
    for _, row in events.iterrows():
        t0 = row["release_time"]
        row_data = row.to_dict()

        # Find the close price at or just after t0
        future = market[market.index >= t0]
        if future.empty:
            for w in windows_minutes:
                row_data[f"ret_{w}m"] = np.nan
            results.append(row_data)
            continue

        p0 = future.iloc[0]["close"]

        for w in windows_minutes:
            t_end = t0 + pd.Timedelta(minutes=w)
            window_slice = market[(market.index >= t0) & (market.index <= t_end)]
            if window_slice.empty:
                row_data[f"ret_{w}m"] = np.nan
            else:
                p_end = window_slice.iloc[-1]["close"]
                row_data[f"ret_{w}m"] = (p_end - p0) / p0

        results.append(row_data)

    return pd.DataFrame(results)
