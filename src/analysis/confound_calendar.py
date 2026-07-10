"""
US macro confound calendar — FOMC decision days and NFP (non-farm payrolls)
release days, 2019-2025.

PBOC events that land on (or the day before/after) a major US
macro release are contaminated: any CNH/rate/equity move that day is
plausibly driven by the Fed/US data, not the Chinese-language signal
being tested. Excluding these lets the remaining sample isolate the
PBOC-specific effect.

FOMC dates are the second day of each 2-day meeting (rate decision +
press conference). NFP dates are the first Friday of each month (the
standard release schedule; a handful of months shift to the second
Friday when the first falls on/adjacent to a holiday, which this
approximation does not special-case — precise to within a few days,
which is enough for a same-day exclusion filter).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

FOMC_DATES = [
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025 (scheduled)
    "2025-01-29", "2025-03-19",
]


def _first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # Friday = weekday 4
    return d + timedelta(days=offset)


def nfp_dates(start_year: int = 2019, end_year: int = 2025) -> list[str]:
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(_first_friday(year, month).isoformat())
    return dates


def confound_dates(pad_days: int = 1) -> set:
    """All FOMC/NFP dates plus a +/- pad_days buffer, as a set of date objects."""
    all_dates = FOMC_DATES + nfp_dates()
    out = set()
    for d in all_dates:
        base = pd.Timestamp(d).date()
        for offset in range(-pad_days, pad_days + 1):
            out.add(base + timedelta(days=offset))
    return out


def filter_confounded(events: pd.DataFrame, date_col: str = "event_date",
                      pad_days: int = 1) -> pd.DataFrame:
    """Drop events whose date falls within pad_days of a FOMC/NFP release."""
    confounds = confound_dates(pad_days=pad_days)
    dates = pd.to_datetime(events[date_col]).dt.date
    mask = ~dates.isin(confounds)
    return events[mask].copy()


if __name__ == "__main__":
    confounds = confound_dates(pad_days=1)
    print(f"{len(confounds)} confound dates (FOMC +/- 1 day, NFP +/- 1 day) covering 2019-2025")
