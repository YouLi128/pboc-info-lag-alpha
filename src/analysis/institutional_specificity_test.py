"""
H3 test: does the same PBOC signal-extraction toolkit (3-class LLM /
dictionary / continuous LLM) find anything on NDRC or MOF text, across
all four market channels, on either a direction test (Granger) or a
magnitude test (contemporaneous volatility regression)?

A clean null here is informative, not a failure: it's consistent with
H1's effect being tied to PBOC's specific institutional role in setting
monetary and liquidity conditions, rather than a generic "any government
text predicts any market" phenomenon.

Verified 2026-07-20 (full rerun after finding a stale-baseline bug in the
H2 coverage test, to rule out the same issue here): NDRC + MOF x all
three scoring methods x all four market channels, 7 candidates crossed
p<0.05, every one failed at least one of the four validation gates
(placebo / Spearman / top-5%-outlier-exclusion / COVID-exclusion). 0 of 7
passed all four. No baseline-bug issue found here — this rerun confirmed
the earlier (dictionary/continuous-only) result rather than changing it.

Usage:
    python -m src.analysis.institutional_specificity_test
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.event_study_pipeline import build_events, align_hourly, run_granger_daily
from src.analysis.dictionary_event_study import build_events_dict
from src.analysis.continuous_event_study import build_events_continuous
from src.analysis.repo_rate_event_study import align_daily_rate
from src.analysis.cgb_yield_event_study import align_daily_cgb
from src.analysis.ndrc_event_study import align_daily_csi300
from src.analysis.volatility_channel_test import volatility_regression, placebo_test_volatility

logger = logging.getLogger(__name__)

COVID_START, COVID_END = pd.Timestamp("2020-02-01", tz="UTC"), pd.Timestamp("2020-04-30", tz="UTC")

SOURCES = {
    "ndrc": {
        "3-class": lambda: build_events("data/processed/ndrc_classified.csv"),
        "dictionary": lambda: build_events_dict("data/processed/ndrc_fg_fulltext.csv", source="ndrc"),
        "continuous": lambda: build_events_continuous("data/processed/ndrc_fg_continuous_scored.csv"),
    },
    "mof": {
        "3-class": lambda: build_events("data/processed/mof_classified.csv"),
        "dictionary": lambda: build_events_dict("data/processed/mof_fg_fulltext.csv", source="mof"),
        "continuous": lambda: build_events_continuous("data/processed/mof_fg_continuous_scored.csv"),
    },
}

MARKETS = [
    ("CNH", lambda ev: align_hourly(ev, "data/raw/cnh_1h_all_events.csv"), [f"ret_{w}h" for w in [1, 2, 4, 8, 24]]),
    ("FDR007", lambda ev: align_daily_rate(ev, "data/raw/repo_rate_daily.csv"), [f"chg_{w}d_bps" for w in [0, 1, 2]]),
    ("CGB10y", lambda ev: align_daily_cgb(ev, "data/raw/cgb_yield_daily.csv"), [f"chg_{w}d_bps" for w in [0, 1, 2]]),
    ("CSI300", lambda ev: align_daily_csi300(ev, "data/raw/csi300_daily.csv"), [f"ret_{w}d" for w in [0, 1, 2]]),
]


def placebo_direction(events: pd.DataFrame, align_fn, y_col: str, n_permutations: int = 200, seed: int = 42):
    real_aligned = align_fn(events)
    real = run_granger_daily(real_aligned, y_col=y_col, x_col="surprise")
    real_p = real.get("min_pval", np.nan)
    rng = np.random.default_rng(seed)
    dates = events["release_utc"].to_numpy()
    null_p = []
    for _ in range(n_permutations):
        shuffled = events.copy()
        shuffled["release_utc"] = rng.permutation(dates)
        null_p.append(run_granger_daily(align_fn(shuffled), y_col=y_col, x_col="surprise").get("min_pval", np.nan))
    null_p = np.array(null_p)
    null_p = null_p[~np.isnan(null_p)]
    frac = (null_p <= real_p).mean() if len(null_p) else np.nan
    return real_p, frac, real


def best_lag(granger_res: dict) -> int | None:
    lags = {k: v for k, v in granger_res.items() if k.startswith("lag_") and k.endswith("_pval")}
    return int(min(lags, key=lags.get).split("_")[1]) if lags else None


def spearman_at_lag(aligned: pd.DataFrame, y_col: str, x_col: str, lag: int):
    df = aligned[[x_col, y_col]].dropna().reset_index(drop=True)
    x_lag = df[x_col].shift(lag)
    sub = pd.DataFrame({"x": x_lag, "y": df[y_col]}).dropna()
    if len(sub) < 5:
        return np.nan, np.nan
    r = stats.spearmanr(sub["x"], sub["y"])
    return r.statistic, r.pvalue


def top5pct_excluded(aligned: pd.DataFrame) -> pd.DataFrame:
    thresh = aligned["surprise"].abs().quantile(0.95)
    return aligned[aligned["surprise"].abs() < thresh].copy()


def covid_excluded(aligned: pd.DataFrame) -> pd.DataFrame:
    return aligned[~aligned["release_utc"].between(COVID_START, COVID_END)].copy()


def gate_check_direction(events: pd.DataFrame, align_fn, y_col: str, label: str) -> bool:
    aligned = align_fn(events)
    real_p, frac, real_res = placebo_direction(events, align_fn, y_col)
    lag = best_lag(real_res)
    placebo_ok = frac < 0.05
    if lag:
        _, sp = spearman_at_lag(aligned, y_col, "surprise", lag)
        spearman_ok = sp < 0.05 if not np.isnan(sp) else False
    else:
        spearman_ok = False
    p_ex = run_granger_daily(top5pct_excluded(aligned), y_col=y_col, x_col="surprise").get("min_pval", np.nan)
    p_cv = run_granger_daily(covid_excluded(aligned), y_col=y_col, x_col="surprise").get("min_pval", np.nan)
    all_pass = placebo_ok and spearman_ok and (p_ex < 0.05) and (p_cv < 0.05)
    print(f"    [DIR] {label}: p={real_p:.4f} placebo={frac*100:.1f}% "
          f"spearman={'PASS' if spearman_ok else 'FAIL'} "
          f"excl5%={'PASS' if p_ex < 0.05 else 'FAIL'}(p={p_ex:.3f}) "
          f"exclCOVID={'PASS' if p_cv < 0.05 else 'FAIL'}(p={p_cv:.3f}) "
          f"=> {'*** ALL 4 GATES PASS ***' if all_pass else 'rejected'}")
    return all_pass


def gate_check_volatility(events: pd.DataFrame, align_fn, y_col: str, label: str) -> bool:
    aligned = align_fn(events)
    real = volatility_regression(aligned, y_col)
    if "error" in real:
        return False
    real_p, frac = placebo_test_volatility(events, lambda ev, _: align_fn(ev), None, y_col)
    spearman_ok = real["spearman_pval"] < 0.05
    p_ex = volatility_regression(top5pct_excluded(aligned), y_col).get("ols_pval", np.nan)
    p_cv = volatility_regression(covid_excluded(aligned), y_col).get("ols_pval", np.nan)
    all_pass = (frac < 0.05) and spearman_ok and (p_ex < 0.05) and (p_cv < 0.05)
    print(f"    [VOL] {label}: p={real['ols_pval']:.4f} placebo={frac*100:.1f}% "
          f"spearman={'PASS' if spearman_ok else 'FAIL'}(p={real['spearman_pval']:.3f}) "
          f"excl5%={'PASS' if p_ex < 0.05 else 'FAIL'}(p={p_ex:.3f}) "
          f"exclCOVID={'PASS' if p_cv < 0.05 else 'FAIL'}(p={p_cv:.3f}) "
          f"=> {'*** ALL 4 GATES PASS ***' if all_pass else 'rejected'}")
    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    validated = []
    for corpus, methods in SOURCES.items():
        for method_name, build_fn in methods.items():
            events = build_fn()
            print(f"\n{'='*72}\n{corpus.upper()} / {method_name} ({len(events)} events)\n{'='*72}")
            for mkt_name, align_fn, cols in MARKETS:
                print(f"  --- {mkt_name} ---")
                for col in cols:
                    aligned = align_fn(events)
                    dp = run_granger_daily(aligned, y_col=col, x_col="surprise").get("min_pval", np.nan)
                    vp = volatility_regression(aligned, col).get("ols_pval", np.nan)
                    flags = ("" if dp >= 0.05 else " DIR<0.05") + ("" if vp >= 0.05 else " VOL<0.05")
                    print(f"    {col}: dir_p={dp:.4f} vol_p={vp:.4f}{flags}")
                    label = f"{corpus}/{method_name}/{mkt_name}/{col}"
                    if dp < 0.05 and gate_check_direction(events, align_fn, col, label):
                        validated.append(("DIR", label))
                    if vp < 0.05 and gate_check_volatility(events, align_fn, col, label):
                        validated.append(("VOL", label))

    print(f"\n\n{'#'*72}\nFINAL: {len(validated)} candidates passed all 4 gates\n{'#'*72}")
    for kind, label in validated:
        print(f"  {kind}: {label}")
