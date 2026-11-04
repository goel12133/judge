"""Nuisance regressors and residualisation.

Every probe is fit on residuals of the response after regressing out, within training
folds only, the features a language model trivially shares with the reader:
  surprisal (and spillover: surprisal of the previous k words), log frequency, word
  length, position in story, and the same lags for frequency and length.

The headline number for a channel is the probe's correlation with these residuals,
not with the raw response. Residualisation must be fit on training items and applied to
test items (no peeking); `Residualiser` enforces that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

NUISANCE = ["surprisal", "log_freq", "length", "position"]


def add_lags(df: pd.DataFrame, cols=("surprisal", "log_freq", "length"), k: int = 2,
             group: str = "item") -> pd.DataFrame:
    """Add lag-1..k copies of `cols` within each story (spillover regressors)."""
    out = df.copy()
    for c in cols:
        for lag in range(1, k + 1):
            out[f"{c}_lag{lag}"] = out.groupby(group)[c].shift(lag)
    return out


def nuisance_columns(k: int = 2) -> list[str]:
    cols = list(NUISANCE)
    for c in ("surprisal", "log_freq", "length"):
        cols += [f"{c}_lag{lag}" for lag in range(1, k + 1)]
    return cols


class Residualiser:
    """Fit a linear nuisance model on training rows; return residuals for any rows."""

    def __init__(self, cols: list[str] | None = None):
        self.cols = cols or nuisance_columns()
        self.model = LinearRegression()
        self.means_: pd.Series | None = None

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.cols].astype(float)
        X = X.fillna(self.means_)
        return X.values

    def fit(self, df: pd.DataFrame, y: str) -> "Residualiser":
        self.means_ = df[self.cols].astype(float).mean()
        self.model.fit(self._X(df), df[y].values)
        return self

    def residuals(self, df: pd.DataFrame, y: str) -> np.ndarray:
        return df[y].values - self.model.predict(self._X(df))

    def r2(self, df: pd.DataFrame, y: str) -> float:
        yv = df[y].values
        res = self.residuals(df, y)
        return float(1 - res.var() / yv.var())
