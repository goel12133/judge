"""Linear probes with blocked cross-validation and the controls the paper commits to.

- Folds are by story (`group`), never random over segments: responses are autocorrelated.
- Ridge alpha is chosen by inner CV on the training stories only.
- Residualisation (baselines.Residualiser) is fit inside each training fold.
- Controls: shuffled labels (label permutation within story) and a random-feature
  control (features replaced by Gaussian noise of the same shape), both run through the
  identical pipeline. A real probe must beat both.
- Uncertainty: bootstrap over stories.

`features` is an (n_items, d) array aligned to `df` rows: hidden states from a model,
or any feature set. With no language model available, `features` can be any candidate
representation; the pipeline is the same.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from .baselines import Residualiser


@dataclass
class ProbeResult:
    r_raw: float          # corr(pred, raw y) on held-out items (reported but not headline)
    r_resid: float        # corr(pred, residual y) on held-out items — headline
    per_fold: list[float] # r_resid per held-out story fold
    n_items: int
    n_folds: int
    alpha: float | None = None
    preds: np.ndarray | None = None   # held-out predictions (for story bootstrap without refit)
    resid: np.ndarray | None = None   # matching held-out residual targets


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def blocked_probe(df: pd.DataFrame, features: np.ndarray, y: str, group: str = "item",
                  nuisance_cols: list[str] | None = None,
                  alphas=(0.1, 1, 10, 100, 1000, 10000), n_splits: int | None = None,
                  residualise: bool = True) -> ProbeResult:
    """Ridge probe from `features` to residual `y`, leave-stories-out."""
    groups = df[group].values
    n_groups = len(np.unique(groups))
    n_splits = n_splits or n_groups
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    preds = np.zeros(len(df))
    resid_all = np.zeros(len(df))
    per_fold = []
    alpha_used = []
    for tr, te in gkf.split(features, groups=groups):
        dtr, dte = df.iloc[tr], df.iloc[te]
        if residualise:
            res = Residualiser(nuisance_cols).fit(dtr, y)
            ytr = res.residuals(dtr, y)
            yte = res.residuals(dte, y)
        else:
            ytr, yte = dtr[y].values, dte[y].values
        mu, sd = features[tr].mean(0), features[tr].std(0) + 1e-8
        Xtr, Xte = (features[tr] - mu) / sd, (features[te] - mu) / sd
        model = RidgeCV(alphas=alphas).fit(Xtr, ytr)
        p = model.predict(Xte)
        preds[te] = p
        resid_all[te] = yte
        per_fold.append(_corr(p, yte))
        alpha_used.append(float(model.alpha_))
    return ProbeResult(
        r_raw=_corr(preds, df[y].values),
        r_resid=_corr(preds, resid_all),
        per_fold=per_fold,
        n_items=len(df),
        n_folds=len(per_fold),
        alpha=float(np.median(alpha_used)),
        preds=preds, resid=resid_all,
    )


def shuffled_label_control(df: pd.DataFrame, features: np.ndarray, y: str, seed: int = 0,
                           group: str = "item", **kw) -> ProbeResult:
    """Permute y within each story; probe should fall to ~0."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d[y] = d.groupby(group)[y].transform(lambda s: rng.permutation(s.values))
    return blocked_probe(d, features, y, group=group, **kw)


def random_feature_control(df: pd.DataFrame, features: np.ndarray, y: str, seed: int = 0,
                           **kw) -> ProbeResult:
    """Replace features with Gaussian noise of the same shape; probe should fall to ~0."""
    rng = np.random.default_rng(seed)
    return blocked_probe(df, rng.standard_normal(features.shape), y, **kw)


def bootstrap_over_groups(df: pd.DataFrame, features: np.ndarray, y: str, n_boot: int = 200,
                          seed: int = 0, group: str = "item", **kw) -> dict:
    """Bootstrap r_resid by resampling stories with replacement (items within a story
    stay together). Returns median and 95% interval."""
    rng = np.random.default_rng(seed)
    groups = df[group].unique()
    rs = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        # Build a resampled frame; duplicate stories get distinct group ids so folds work.
        parts, feats = [], []
        for i, g in enumerate(pick):
            m = df[group].values == g
            part = df[m].copy()
            part[group] = f"{g}_{i}"
            parts.append(part)
            feats.append(features[m])
        d = pd.concat(parts, ignore_index=True)
        X = np.vstack(feats)
        if d[group].nunique() < 3:
            continue
        rs.append(blocked_probe(d, X, y, group=group, **kw).r_resid)
    rs = np.array(rs)
    return {"median": float(np.median(rs)), "ci95": (float(np.percentile(rs, 2.5)),
                                                     float(np.percentile(rs, 97.5))),
            "n_boot": int(len(rs))}


def bootstrap_heldout(groups: np.ndarray, preds: np.ndarray, resid: np.ndarray,
                      n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the held-out correlation by resampling stories with replacement over
    already-held-out (prediction, residual) pairs. No refitting: the probes are fixed, the
    uncertainty is over which stories were measured. Returns median and 95% interval."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        sel = np.concatenate([idx[g] for g in pick])
        rs.append(_corr(preds[sel], resid[sel]))
    rs = np.array(rs)
    return {"median": float(np.median(rs)), "ci95": (float(np.percentile(rs, 2.5)),
                                                     float(np.percentile(rs, 97.5))),
            "n_boot": int(len(rs)), "method": "resample stories over held-out predictions"}
