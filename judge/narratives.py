"""Narratives (OpenNeuro ds002345, CC0): engagement channel.

Inputs are the per-story files written by scripts/narratives_prepare_colab.py:
    narr_{story}.npz with bold (n_subj, n_TR, n_parcels) z-scored per parcel, the Gentle word
    alignment in scan time, TR and stimulus onset.

Three things live here:
  * word -> TR alignment: a per-word feature matrix (n_words, d) becomes a per-TR design
    (n_TR, d) by placing each word's vector at its onset (impulse, sampled on a fine grid),
    convolving with a canonical double-gamma HRF (Glover 1999) and sampling at the TRs.
    An extra scalar lag can shift the HRF; it is chosen on a dev story (DESIGN.md, control 3).
  * inter-subject reliability: split-half ISC per parcel (Spearman-Brown corrected), the
    ceiling for a probe to the group-mean time series; ISC itself as a parcel weight.
  * a loader that returns the story's words (for hidden-state extraction) and targets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def glover_hrf(t: np.ndarray) -> np.ndarray:
    """Canonical double-gamma HRF sampled at times t (s); peak ~5 s, undershoot ~15 s.
    Glover 1999 parameters as in SPM/nilearn; scaled to unit peak."""
    from scipy.stats import gamma
    t = np.asarray(t, float)
    h = gamma.pdf(t, 6) - 1 / 6 * gamma.pdf(t, 16)
    h[t < 0] = 0
    return h / h.max()


def words_to_trs(onsets_s: np.ndarray, feats: np.ndarray, n_tr: int, tr: float,
                 lag_s: float = 0.0, dt: float = 0.1, hrf: bool = True) -> np.ndarray:
    """Place each word's feature vector at its onset on a dt grid, convolve with the HRF
    (or box of one grid step if hrf=False), sample at TR midpoints. Returns (n_tr, d).
    Words whose onset falls after the scan is over are dropped."""
    onsets_s = np.asarray(onsets_s, float)
    feats = np.asarray(feats, float)
    if feats.ndim == 1:
        feats = feats[:, None]
    n_grid = int(np.ceil(n_tr * tr / dt)) + int(32 / dt)
    grid = np.zeros((n_grid, feats.shape[1]))
    idx = np.round((onsets_s + lag_s) / dt).astype(int)
    ok = (idx >= 0) & (idx < n_grid)
    np.add.at(grid, idx[ok], feats[ok])
    if hrf:
        kern = glover_hrf(np.arange(0, 32, dt))
        grid = np.apply_along_axis(lambda c: np.convolve(c, kern)[: n_grid], 0, grid)
    sample = np.round((np.arange(n_tr) * tr + tr / 2) / dt).astype(int)
    return grid[sample]


def split_half_isc(bold: np.ndarray, n_splits: int = 50, seed: int = 0) -> np.ndarray:
    """Per-parcel reliability of the group-mean time series: correlation between the mean
    of two random halves of subjects, Spearman-Brown corrected, averaged over splits.
    bold: (n_subj, n_TR, n_parcels). Returns (n_parcels,)."""
    rng = np.random.default_rng(seed)
    n = bold.shape[0]
    out = np.zeros((n_splits, bold.shape[2]))
    for i in range(n_splits):
        perm = rng.permutation(n)
        a, b = bold[perm[: n // 2]].mean(0), bold[perm[n // 2:]].mean(0)
        a = (a - a.mean(0)) / (a.std(0) + 1e-9); b = (b - b.mean(0)) / (b.std(0) + 1e-9)
        r = (a * b).mean(0)
        out[i] = 2 * r / (1 + r)
    return out.mean(0)


def leave_one_out_isc(bold: np.ndarray) -> np.ndarray:
    """Classic ISC: each subject vs the mean of the others, averaged. (n_parcels,)."""
    n = bold.shape[0]
    rs = []
    for s in range(n):
        others = np.delete(bold, s, 0).mean(0)
        a = bold[s].astype(float); a = (a - a.mean(0)) / (a.std(0) + 1e-9)
        b = (others - others.mean(0)) / (others.std(0) + 1e-9)
        rs.append((a * b).mean(0))
    return np.mean(rs, 0)


@dataclass
class Story:
    name: str
    bold: np.ndarray          # (n_subj, n_TR, n_parcels) float32
    tr: float
    onset_s: float
    words: np.ndarray         # (n_words,) str
    start_s: np.ndarray       # word onsets in scan time
    end_s: np.ndarray
    text: str
    labels: list[str]

    @property
    def group_mean(self) -> np.ndarray:
        return self.bold.mean(0)

    @property
    def n_tr(self) -> int:
        return self.bold.shape[1]


def load_story(path: str | Path) -> Story:
    z = np.load(path, allow_pickle=False)
    return Story(name=Path(path).stem.replace("narr_", ""), bold=z["bold"].astype(np.float32),
                 tr=float(z["TR"]), onset_s=float(z["onset_s"]), words=z["word"].astype(str),
                 start_s=z["start_s"].astype(float), end_s=z["end_s"].astype(float),
                 text=str(z["text"]), labels=[str(l) for l in z["labels"]])


def network_of(label: str) -> str:
    """Schaefer label '7Networks_LH_Default_PFC_1' -> 'Default'."""
    parts = label.split("_")
    return parts[2] if len(parts) > 2 else label
