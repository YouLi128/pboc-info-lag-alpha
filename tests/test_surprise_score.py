"""Unit tests for surprise score construction."""

import pandas as pd
import pytest

from src.signal_construction.surprise_score import (
    build_event_table,
    compute_surprise,
    encode_stance,
)


def test_encode_stance():
    assert encode_stance("dovish") == -1.0
    assert encode_stance("hawkish") == 1.0
    assert encode_stance("neutral") == 0.0
    assert encode_stance("unknown") == 0.0


def test_build_event_table_basic():
    records = [
        {
            "release_time": "2024-01-15 10:00:00+00:00",
            "segment_type": "forward_guidance",
            "stance": "dovish",
            "confidence": 0.9,
        },
        {
            "release_time": "2024-02-20 10:00:00+00:00",
            "segment_type": "descriptive",
            "stance": "neutral",
            "confidence": 0.8,
        },
    ]
    df = build_event_table(records)
    assert len(df) == 2
    assert df.loc[0, "is_forward_guidance"] is True
    assert df.loc[1, "is_forward_guidance"] is False
    # Non-forward-guidance rows should have weighted_score == 0
    assert df.loc[1, "weighted_score"] == pytest.approx(0.0)


def test_compute_surprise_shape():
    records = [
        {
            "release_time": f"2024-0{i}-01 10:00:00+00:00",
            "segment_type": "forward_guidance",
            "stance": "hawkish",
            "confidence": 0.7,
        }
        for i in range(1, 6)
    ]
    events = build_event_table(records)
    result = compute_surprise(events, baseline_window=3)
    assert "surprise" in result.columns
    assert len(result) == len(events)
