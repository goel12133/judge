"""Suspense and surprise as functions of a belief process (Ely, Frankel & Kamenica 2015).

EFK define, for a belief process mu_t over a finite set of outcomes:
  surprise_t  = d(mu_t, mu_{t-1})                     realised change in beliefs
  suspense_t  = sqrt( E_t[ d(mu_{t+1}, mu_t)^2 ] )    expected magnitude of next change
with d a distance on the simplex (they use squared Euclidean inside the expectation).
These are properties of the belief *process*, not of the text, so they can be computed
for any agent whose beliefs we can read: a human (via ratings) or a language model (via
its probabilities). That is what makes them a test of whether a model's uncertainty is
the *reader's* uncertainty.

Two ways to get a model's belief process over a story:
  (a) outcome-level: define a small set of story outcomes (e.g. "the hero escapes" vs
      "is caught") and elicit the model's probability over them after each segment;
  (b) token-level proxy: mu_t = next-token distribution; surprise is then the
      distributional shift between consecutive next-token distributions, and suspense its
      expected value under sampled continuations.
This module is agnostic: it takes belief arrays (T, K) and, for suspense, either the
exact transition structure or Monte-Carlo samples of mu_{t+1}.

VERIFY against the paper before publication: the exact normalisation and whether the
expectation is over the square root or the root over the expectation (we use the latter,
the paper's "sqrt of expected squared distance" form).
"""
from __future__ import annotations

import numpy as np


def _check_simplex(mu: np.ndarray, tol: float = 1e-6) -> None:
    if mu.ndim != 2:
        raise ValueError("beliefs must be (T, K)")
    if np.any(mu < -tol) or np.any(np.abs(mu.sum(1) - 1) > 1e-4):
        raise ValueError("each row of beliefs must be a probability vector")


def surprise(mu: np.ndarray) -> np.ndarray:
    """Realised surprise per step: Euclidean distance between consecutive beliefs.
    Returns length T with surprise[0] = 0."""
    mu = np.asarray(mu, float)
    _check_simplex(mu)
    d = np.linalg.norm(np.diff(mu, axis=0), axis=1)
    return np.concatenate([[0.0], d])


def suspense_from_samples(mu: np.ndarray, next_samples: np.ndarray) -> np.ndarray:
    """Suspense per step from Monte-Carlo samples of the next belief.

    mu: (T, K) current beliefs; next_samples: (T, S, K) S sampled next-step beliefs for
    each t (e.g. from S sampled continuations re-scored by the model). Returns length T;
    the last entry is the expectation over the samples at the last step (if provided).
    """
    mu = np.asarray(mu, float)
    nxt = np.asarray(next_samples, float)
    if nxt.shape[0] != mu.shape[0] or nxt.shape[2] != mu.shape[1]:
        raise ValueError("next_samples must be (T, S, K) matching mu (T, K)")
    sq = ((nxt - mu[:, None, :]) ** 2).sum(2)   # (T, S)
    return np.sqrt(sq.mean(1))


def suspense_exact(mu: np.ndarray, transitions: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Suspense when the next-belief distribution is known exactly.

    transitions: (T, M, K) the M possible next beliefs at each t; probs: (T, M) their
    probabilities (rows sum to 1). suspense_t = sqrt(sum_m p_m ||mu_{t+1,m} - mu_t||^2).
    """
    mu = np.asarray(mu, float)
    tr = np.asarray(transitions, float)
    p = np.asarray(probs, float)
    sq = ((tr - mu[:, None, :]) ** 2).sum(2)   # (T, M)
    return np.sqrt((p * sq).sum(1))


def token_surprise_from_logits(logp: np.ndarray) -> np.ndarray:
    """Token-level proxy: logp is (T, V) next-token log-probs after each token; surprise is
    the Euclidean shift between consecutive next-token distributions. (Not EFK outcome-level
    surprise; a cheap proxy to be reported as such.)"""
    p = np.exp(np.asarray(logp, float))
    p = p / p.sum(1, keepdims=True)
    return surprise(p)


def segment_mean(series: np.ndarray, boundaries: list[int]) -> np.ndarray:
    """Average a per-token series into segments given boundary end indices (exclusive)."""
    out, start = [], 0
    for end in boundaries:
        out.append(float(series[start:end].mean()) if end > start else float("nan"))
        start = end
    return np.array(out)
