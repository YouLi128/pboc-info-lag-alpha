"""
Volatility-channel test: does |surprise| predict |market reaction|
magnitude, rather than signed returns?

Motivation: every signed-return Granger test run tonight (13 source x
market combinations) came back null, and the 3 marginal p<0.05 hits all
failed a placebo/permutation check. Direction may be the wrong lens —
surprising information could raise reaction magnitude/volatility without
a consistent sign (e.g. if the market's prior about direction was mixed).

Standout result (verified 2026-07-10): PBOC |surprise| vs |10y CGB yield
change| 2 trading days later. Pearson r=-0.171, p=0.014 (Spearman
p=0.010, so not an outlier artifact); passes the placebo/permutation
test (1.5% of 200 shuffled-date reruns were at least as significant,
below the 5% bar). Survives excluding the 2020-02~04 COVID-crash window
(p=0.032/0.034) — not driven by that period alone.

The sign is counterintuitive: LARGER measured surprise associates with
SMALLER 2-day yield moves, not larger. The biggest yield moves in the
sample (2020-03-04 COVID crash, 2024-09 stimulus package) occur on days
with near-zero measured surprise — i.e. the biggest real-world bond
moves are NOT the ones this text-based surprise signal flags as
surprising. Two live hypotheses, neither resolved here: (a) the rolling-
baseline "surprise" measure is picking up rhetorical novelty rather than
market-relevant novelty, or (b) large real yield moves are driven by
macro shocks external to that day's specific PBOC statement, while the
statements that ARE textually surprising tend to land in already-calm
periods. Needs further investigation before treating as informative
about H1 — reported here as a validated but unresolved pattern, not a
confirmed effect in the hypothesized direction.

Tested and ruled out: "overshoot then reversal". The two largest-surprise
events show a same-day move that partly unwinds by day 2 (e.g.
2025-01-10's bond-purchase suspension: +2.54bps day 0 -> -1.16bps by
day 2), which suggested the negative relationship might just be that
overshoot artifact and should weaken or flip positive at longer
horizons as the reversal completes. It does not: re-running with
windows out to 10 trading days keeps the correlation negative
throughout (strongest at day 2, r=-0.171; still negative at day 10,
r=-0.124, p=0.076), never flipping sign. The relationship looks more
structural than a short-term overshoot artifact, but the mechanism
remains unresolved.

Usage:
    python -m src.analysis.volatility_channel_test
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

logger = logging.getLogger(__name__)


def volatility_regression(aligned: pd.DataFrame, y_col: str) -> dict:
    sub = aligned[["surprise", y_col]].dropna().copy()
    if len(sub) < 20:
        return {"error": "insufficient_data", "n_obs": len(sub)}
    sub["abs_surprise"] = sub["surprise"].abs()
    sub["abs_ret"] = sub[y_col].abs()
    model = smf.ols("abs_ret ~ abs_surprise", data=sub).fit()
    pearson = stats.pearsonr(sub["abs_surprise"], sub["abs_ret"])
    spearman = stats.spearmanr(sub["abs_surprise"], sub["abs_ret"])
    return {
        "n_obs": len(sub),
        "ols_coef": model.params.get("abs_surprise", np.nan),
        "ols_pval": model.pvalues.get("abs_surprise", np.nan),
        "pearson_r": pearson.statistic,
        "pearson_pval": pearson.pvalue,
        "spearman_r": spearman.statistic,
        "spearman_pval": spearman.pvalue,
    }


def placebo_test_volatility(events: pd.DataFrame, align_fn, market_csv: str,
                            y_col: str, n_permutations: int = 200, seed: int = 42) -> float:
    """Returns the fraction of permuted-date reruns at least as significant as the real one."""
    real_aligned = align_fn(events, market_csv)
    real = volatility_regression(real_aligned, y_col)
    real_p = real["ols_pval"]

    rng = np.random.default_rng(seed)
    dates = events["release_utc"].to_numpy()
    null_p = []
    for _ in range(n_permutations):
        shuffled = events.copy()
        shuffled["release_utc"] = rng.permutation(dates)
        aligned = align_fn(shuffled, market_csv)
        res = volatility_regression(aligned, y_col)
        null_p.append(res.get("ols_pval", np.nan))

    null_p = np.array(null_p)
    null_p = null_p[~np.isnan(null_p)]
    return real_p, (null_p <= real_p).mean()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.analysis.event_study_pipeline import build_events
    from src.analysis.cgb_yield_event_study import align_daily_cgb

    events = build_events("data/processed/corpus_classified.csv")
    aligned = align_daily_cgb(events, "data/raw/cgb_yield_daily.csv")

    print("=== PBOC |surprise| -> |10y CGB yield change| ===")
    for w in [0, 1, 2]:
        col = f"chg_{w}d_bps"
        res = volatility_regression(aligned, col)
        print(f"{col}: {res}")

    real_p, frac = placebo_test_volatility(events, align_daily_cgb, "data/raw/cgb_yield_daily.csv", "chg_2d_bps")
    print(f"\nchg_2d_bps placebo test: real p={real_p:.4f}, "
          f"fraction of shuffled runs at least as significant={frac:.3f}")
