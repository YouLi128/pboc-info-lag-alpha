"""
H2 test: does the surprise signal predict market reactions more strongly
for events NOT covered by English-language wires (GDELT), vs events that
were?

Rather than splitting into two small subsamples and running Granger
causality on each (underpowered with ~100 events per group), this fits
one regression on the full sample with an interaction term:

    return ~ surprise + uncovered + surprise:uncovered

A significant, positive surprise:uncovered coefficient means the surprise
signal's effect on returns is stronger specifically when English-language
media did NOT pick up the story — the cleanest available evidence for the
cross-border information-lag hypothesis (H2).

Full matrix (verified 2026-07-20): reruns this test across all three
surprise-scoring methods (3-class LLM / dictionary / continuous LLM) and
all four PBOC market channels, reusing the existing GDELT coverage flags
(no new API calls). Motivation: H2 was originally only tested with the
3-class stance score, which H1's own matrix showed is the least sensitive
of the three methods. An earlier version of this test reported a near-miss
(CNH 2h, p=0.0055, placebo 5.8%) that turned out to be a byproduct of the
same fixed-event-count baseline bug retracted elsewhere in the project
(see event_study_pipeline.build_events docstring). Rerun on the corrected
60-day rolling baseline across all three methods x four markets (15
windows total), three candidates crossed p<0.05 but every one failed the
placebo test (8.3%-12.0%, all above the 5% bar) — H2 is a clean null.

Usage:
    python -m src.analysis.coverage_split_test
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.analysis.event_study_pipeline import build_events, align_hourly
from src.analysis.dictionary_event_study import build_events_dict
from src.analysis.continuous_event_study import build_events_continuous
from src.analysis.repo_rate_event_study import align_daily_rate
from src.analysis.cgb_yield_event_study import align_daily_cgb
from src.analysis.ndrc_event_study import align_daily_csi300

logger = logging.getLogger(__name__)

COVERAGE_CSV = "data/processed/gdelt_coverage.csv"

SOURCES = {
    "3-class": lambda: build_events("data/processed/corpus_classified.csv"),
    "dictionary": lambda: build_events_dict("data/processed/pboc_fg_fulltext.csv"),
    "continuous": lambda: build_events_continuous("data/processed/pboc_fg_continuous_scored.csv"),
}

MARKETS = [
    ("CNH", lambda ev: align_hourly(ev, "data/raw/cnh_1h_all_events.csv"), [f"ret_{w}h" for w in [1, 2, 4, 8, 24]]),
    ("FDR007", lambda ev: align_daily_rate(ev, "data/raw/repo_rate_daily.csv"), [f"chg_{w}d_bps" for w in [0, 1, 2]]),
    ("CGB10y", lambda ev: align_daily_cgb(ev, "data/raw/cgb_yield_daily.csv"), [f"chg_{w}d_bps" for w in [0, 1, 2]]),
    ("CSI300", lambda ev: align_daily_csi300(ev, "data/raw/csi300_daily.csv"), [f"ret_{w}d" for w in [0, 1, 2]]),
]


def merge_coverage(aligned: pd.DataFrame) -> pd.DataFrame:
    aligned = aligned.copy()
    aligned["release_utc"] = pd.to_datetime(aligned["release_utc"], utc=True)
    coverage = pd.read_csv(COVERAGE_CSV)
    coverage["release_utc"] = pd.to_datetime(coverage["release_utc"], utc=True)
    merged = aligned.merge(coverage[["release_utc", "covered"]], on="release_utc", how="inner")
    merged["uncovered"] = 1 - merged["covered"]
    return merged


def run_interaction(df: pd.DataFrame, y_col: str) -> dict:
    sub = df[["surprise", "uncovered", y_col]].dropna()
    if len(sub) < 20:
        return {"error": "insufficient_data", "n_obs": len(sub)}
    model = smf.ols(f"{y_col} ~ surprise * uncovered", data=sub).fit()
    return {
        "n_obs": len(sub),
        "coef": model.params.get("surprise:uncovered", np.nan),
        "pval": model.pvalues.get("surprise:uncovered", np.nan),
    }


def placebo_interaction(df: pd.DataFrame, y_col: str, n_permutations: int = 300, seed: int = 42) -> tuple[float, float]:
    """Shuffle the covered/uncovered label (keeping surprise and market data fixed)
    and recompute the interaction p-value each time. Returns (real_p, fraction of
    shuffled runs at least as significant as the real result)."""
    sub = df[["surprise", "uncovered", y_col]].dropna().reset_index(drop=True)
    if len(sub) < 20:
        return np.nan, np.nan
    real_p = run_interaction(sub, y_col)["pval"]
    rng = np.random.default_rng(seed)
    uncov = sub["uncovered"].to_numpy()
    null_p = []
    for _ in range(n_permutations):
        shuffled = sub.copy()
        shuffled["uncovered"] = rng.permutation(uncov)
        null_p.append(run_interaction(shuffled, y_col).get("pval", np.nan))
    null_p = np.array(null_p)
    null_p = null_p[~np.isnan(null_p)]
    frac = (null_p <= real_p).mean() if len(null_p) else np.nan
    return real_p, frac


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    coverage = pd.read_csv(COVERAGE_CSV)
    print(f"GDELT coverage: {coverage['covered'].sum()} / {len(coverage)} events "
          f"({100 * coverage['covered'].mean():.0f}%) had English-language wire coverage")

    candidates = []
    for src_name, build_fn in SOURCES.items():
        events = build_fn()
        print(f"\n{'='*70}\nSurprise source: {src_name} ({len(events)} events)\n{'='*70}")
        for mkt_name, align_fn, cols in MARKETS:
            merged = merge_coverage(align_fn(events))
            print(f"\n--- {mkt_name} (n_merged={len(merged)}) ---")
            for col in cols:
                r = run_interaction(merged, col)
                if "error" in r:
                    print(f"  {col}: {r}")
                    continue
                flag = " <<<" if r["pval"] < 0.05 else ""
                print(f"  {col}: n={r['n_obs']} coef={r['coef']:.6f} p={r['pval']:.4f}{flag}")
                if r["pval"] < 0.05:
                    candidates.append((src_name, mkt_name, col, merged))

    print(f"\n\n{'#'*70}\n{len(candidates)} candidates crossed p<0.05 — running placebo test on each\n{'#'*70}")
    for src_name, mkt_name, col, merged in candidates:
        real_p, frac = placebo_interaction(merged, col)
        verdict = "PASS" if frac < 0.05 else "FAIL"
        print(f"{src_name} x {mkt_name} x {col}: real_p={real_p:.4f}, placebo={frac*100:.1f}% -> {verdict}")
