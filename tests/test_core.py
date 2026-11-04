import numpy as np
import pandas as pd
import pytest

from judge import baselines, beliefs, ceiling, probe


def synth(n_stories=8, n_words=120, n_subj=30, seed=0, signal=1.0):
    """Synthetic words with nuisance regressors, a hidden feature, and subject responses."""
    rng = np.random.default_rng(seed)
    rows = []
    hidden = []
    for s in range(n_stories):
        for w in range(n_words):
            surp = rng.gamma(2, 1.5)
            lf = rng.normal(18, 3)
            ln = rng.integers(1, 10)
            h = rng.standard_normal(5)
            hidden.append(h)
            # true mean response: nuisance + hidden signal on first two dims
            mu = 5.7 + 0.03 * surp - 0.01 * lf + 0.02 * ln + signal * 0.05 * (h[0] - h[1])
            rows.append(dict(item=s, zone=w, surprisal=surp, log_freq=lf, length=ln,
                             position=w, mu=mu))
    words = pd.DataFrame(rows)
    H = np.array(hidden)
    long = []
    for subj in range(n_subj):
        noise = rng.normal(0, 0.15, len(words))
        long.append(pd.DataFrame(dict(WorkerId=subj, item=words["item"], zone=words["zone"],
                                      y=words["mu"].values + noise)))
    long = pd.concat(long, ignore_index=True)
    long["key"] = long["item"].astype(str) + ":" + long["zone"].astype(str)
    words["y"] = long.groupby(["item", "zone"])["y"].mean().values
    words = baselines.add_lags(words)
    return words, H, long


def test_ceiling_recovers_reliability():
    words, H, long = synth()
    c = ceiling.split_half_ceiling(long, "WorkerId", "key", "y", n_splits=50)
    assert 0.5 < c["median"] < 1.0
    assert c["n_subjects"] == 30
    # halving noise raises the ceiling
    long2 = long.copy()
    long2["y"] = words.set_index(["item", "zone"]).loc[
        list(zip(long2["item"], long2["zone"]))]["mu"].values + np.random.default_rng(1).normal(0, 0.05, len(long2))
    c2 = ceiling.split_half_ceiling(long2, "WorkerId", "key", "y", n_splits=50)
    assert c2["median"] > c["median"]


def test_residualiser_removes_nuisance():
    words, H, _ = synth()
    r = baselines.Residualiser().fit(words, "y")
    assert r.r2(words, "y") > 0.1
    res = r.residuals(words, "y")
    assert abs(np.corrcoef(res, words["surprisal"].fillna(0))[0, 1]) < 0.05


def test_probe_finds_signal_and_controls_fail():
    words, H, _ = synth()
    real = probe.blocked_probe(words, H, "y")
    shuf = probe.shuffled_label_control(words, H, "y")
    rand = probe.random_feature_control(words, H, "y")
    assert real.r_resid > 0.3
    assert abs(shuf.r_resid) < 0.15
    assert abs(rand.r_resid) < 0.15
    assert real.n_folds == 8


def test_probe_no_signal_is_near_zero():
    words, H, _ = synth(signal=0.0)
    real = probe.blocked_probe(words, H, "y")
    assert abs(real.r_resid) < 0.15


def test_bootstrap_interval_contains_point():
    words, H, _ = synth(n_stories=6)
    pt = probe.blocked_probe(words, H, "y").r_resid
    b = probe.bootstrap_over_groups(words, H, "y", n_boot=20)
    assert b["ci95"][0] - 0.2 <= pt <= b["ci95"][1] + 0.2


def test_surprise_and_suspense_hand_case():
    # Two outcomes. Beliefs move 0.5 -> 0.9 -> 0.9.
    mu = np.array([[0.5, 0.5], [0.9, 0.1], [0.9, 0.1]])
    s = beliefs.surprise(mu)
    assert s[0] == 0
    assert np.isclose(s[1], np.sqrt(0.4**2 + 0.4**2))
    assert s[2] == 0
    # From (0.5,0.5), next belief is (1,0) or (0,1) with prob 0.5 each:
    # suspense = sqrt(0.5*0.5 + 0.5*0.5) = sqrt(0.5)
    tr = np.array([[[1, 0], [0, 1]]], float)
    sp = beliefs.suspense_exact(mu[:1], tr, np.array([[0.5, 0.5]]))
    assert np.isclose(sp[0], np.sqrt(0.5))
    # Resolved belief has zero suspense
    tr2 = np.array([[[0.9, 0.1], [0.9, 0.1]]], float)
    assert np.isclose(beliefs.suspense_exact(mu[1:2], tr2, np.array([[0.5, 0.5]]))[0], 0)


def test_suspense_from_samples_matches_exact():
    rng = np.random.default_rng(0)
    mu = np.array([[0.5, 0.5]])
    samples = np.array([[[1, 0]] * 5000 + [[0, 1]] * 5000], float)
    sp = beliefs.suspense_from_samples(mu, samples)
    assert np.isclose(sp[0], np.sqrt(0.5), atol=1e-6)


def test_beliefs_reject_non_simplex():
    with pytest.raises(ValueError):
        beliefs.surprise(np.array([[0.5, 0.6]]))


def test_segment_mean():
    out = beliefs.segment_mean(np.arange(6, dtype=float), [2, 4, 6])
    assert list(out) == [0.5, 2.5, 4.5]
