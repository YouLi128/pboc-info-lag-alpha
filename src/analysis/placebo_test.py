"""
Placebo test for the CNH event study.

Directly addresses the "signal is just noise" concern: if the surprise
score carries real information, Granger causality on the *actual* event
dates should show a stronger (lower p-value) relationship than on
randomly relabeled event dates. Re-runs the CNH hourly alignment many
times with event dates shuffled (surprise values kept, dates randomly
reassigned from the same pool of possible trading hours), and compares
the real min p-value against the resulting null distribution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analysis.event_study_pipeline import align_hourly, build_events, run_granger_daily

logger = logging.getLogger(__name__)


def placebo_pvalues(events: pd.DataFrame, cnh_1h_csv: str,
                    y_col: str = "ret_1h", n_permutations: int = 200,
                    seed: int = 42) -> np.ndarray:
    """
    Shuffle release_utc across events (keeping surprise/stance fixed) and
    recompute the Granger min p-value each time. Returns an array of
    n_permutations null min-p-values.
    """
    rng = np.random.default_rng(seed)
    dates = events["release_utc"].to_numpy()
    results = []

    for i in range(n_permutations):
        shuffled = events.copy()
        shuffled["release_utc"] = rng.permutation(dates)
        aligned = align_hourly(shuffled, cnh_1h_csv)
        res = run_granger_daily(aligned, y_col=y_col, x_col="surprise")
        results.append(res.get("min_pval", np.nan))
        if (i + 1) % 50 == 0:
            logger.info("  permutation %d / %d done", i + 1, n_permutations)

    return np.array(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = build_events("data/processed/corpus_classified.csv")
    aligned_real = align_hourly(events, "data/raw/cnh_1h_all_events.csv")

    print("=== Real event dates ===")
    for w in [1, 2, 4, 8, 24]:
        col = f"ret_{w}h"
        real = run_granger_daily(aligned_real, y_col=col, x_col="surprise")
        real_min_p = real.get("min_pval", np.nan)

        logger.info("Running %d placebo permutations for %s …", 200, col)
        null_p = placebo_pvalues(events, "data/raw/cnh_1h_all_events.csv", y_col=col, n_permutations=200)
        null_p = null_p[~np.isnan(null_p)]

        frac_better = (null_p <= real_min_p).mean() if len(null_p) else np.nan
        print(f"\n--- {col} ---")
        print(f"real min p-value:         {real_min_p:.4f}")
        print(f"placebo min p-value mean: {np.nanmean(null_p):.4f}  (n={len(null_p)})")
        print(f"fraction of placebo runs at least as significant as real: {frac_better:.3f}")
        print("(this fraction is the permutation-test p-value; <0.05 = real dates beat noise floor)")
