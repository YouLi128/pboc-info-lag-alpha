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

Usage:
    python -m src.analysis.coverage_split_test
"""

from __future__ import annotations

import logging

import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

COVERAGE_CSV = "data/processed/gdelt_coverage.csv"
CNH_CSV = "data/processed/events_aligned_hourly_full.csv"
REPO_CSV = "data/processed/events_aligned_repo_rate.csv"


def load_merged(aligned_csv: str) -> pd.DataFrame:
    aligned = pd.read_csv(aligned_csv, parse_dates=["release_utc"])
    coverage = pd.read_csv(COVERAGE_CSV, parse_dates=["release_utc"])
    merged = aligned.merge(coverage[["release_utc", "covered"]], on="release_utc", how="inner")
    merged["uncovered"] = 1 - merged["covered"]
    return merged


def run_interaction(df: pd.DataFrame, y_col: str) -> None:
    sub = df[["surprise", "uncovered", y_col]].dropna()
    if len(sub) < 20:
        print(f"  {y_col}: insufficient data (n={len(sub)})")
        return
    model = smf.ols(f"{y_col} ~ surprise * uncovered", data=sub).fit()
    coef = model.params.get("surprise:uncovered", float("nan"))
    pval = model.pvalues.get("surprise:uncovered", float("nan"))
    n_covered = (sub["uncovered"] == 0).sum()
    n_uncovered = (sub["uncovered"] == 1).sum()
    print(f"  {y_col}: n={len(sub)} (covered={n_covered}, uncovered={n_uncovered}) | "
          f"surprise:uncovered coef={coef:.6f}  p={pval:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    coverage = pd.read_csv(COVERAGE_CSV)
    print(f"GDELT coverage: {coverage['covered'].sum()} / {len(coverage)} events "
          f"({100 * coverage['covered'].mean():.0f}%) had English-language wire coverage\n")

    print("=== CNH hourly returns: surprise x uncovered interaction ===")
    cnh = load_merged(CNH_CSV)
    for w in [1, 2, 4, 8, 24]:
        run_interaction(cnh, f"ret_{w}h")

    print("\n=== FDR007 repo-rate changes: surprise x uncovered interaction ===")
    repo = load_merged(REPO_CSV)
    for w in [0, 1, 2]:
        run_interaction(repo, f"chg_{w}d_bps")
