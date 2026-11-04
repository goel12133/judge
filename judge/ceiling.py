"""Inter-subject noise ceilings.

A probe's correlation with a per-item mean response is bounded by how reliable that
mean is. We estimate the ceiling with split-half reliability across subjects, corrected
with Spearman–Brown to the full subject count, repeated over random splits.

For a channel measured on N subjects:
  r_half = corr(mean over half A, mean over half B) per split
  r_full = 2 r_half / (1 + r_half)        (Spearman–Brown)
Reported numbers: median and bootstrap interval over splits. A probe at r on the
subject-mean response is reported as r / r_full ("fraction of ceiling").
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_half_ceiling(long: pd.DataFrame, subject: str, item: str, value: str,
                       n_splits: int = 200, seed: int = 0, min_n: int = 2) -> dict:
    """Split-half reliability of per-item means in a long table (one row per subject×item).

    Items with fewer than `min_n` subjects in either half are dropped for that split.
    Returns dict with r_half (array), r_full (array), median, ci95 (2.5, 97.5), n_items.
    """
    rng = np.random.default_rng(seed)
    subjects = long[subject].unique()
    if len(subjects) < 2:
        raise ValueError("need at least two subjects")
    r_half = []
    n_items_used = []
    for _ in range(n_splits):
        perm = rng.permutation(subjects)
        a = set(perm[: len(perm) // 2])
        ma = long[long[subject].isin(a)].groupby(item)[value].agg(["mean", "size"])
        mb = long[~long[subject].isin(a)].groupby(item)[value].agg(["mean", "size"])
        j = ma.join(mb, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j["size_a"] >= min_n) & (j["size_b"] >= min_n)]
        if len(j) < 3:
            continue
        r = np.corrcoef(j["mean_a"], j["mean_b"])[0, 1]
        r_half.append(r)
        n_items_used.append(len(j))
    r_half = np.array(r_half)
    r_full = 2 * r_half / (1 + r_half)
    return {
        "r_half": r_half,
        "r_full": r_full,
        "median": float(np.median(r_full)),
        "ci95": (float(np.percentile(r_full, 2.5)), float(np.percentile(r_full, 97.5))),
        "n_items": int(np.median(n_items_used)) if n_items_used else 0,
        "n_subjects": int(len(subjects)),
    }


def fraction_of_ceiling(r: float, ceiling: float) -> float:
    if ceiling <= 0:
        return float("nan")
    return float(r / ceiling)
