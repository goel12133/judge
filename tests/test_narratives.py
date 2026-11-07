import numpy as np

from judge import narratives as nr


def test_hrf_shape_and_peak():
    t = np.arange(0, 32, 0.1)
    h = nr.glover_hrf(t)
    assert abs(h.max() - 1) < 1e-9
    assert 4 < t[h.argmax()] < 7          # peak ~5 s
    assert h[t > 25].max() < 0.05         # back to baseline


def test_words_to_trs_recovers_a_delayed_signal():
    # one scalar feature pulsing at known onsets; the TR series should peak ~5 s later
    tr, n_tr = 1.5, 60
    onsets = np.array([10.0, 40.0])
    feats = np.array([1.0, 2.0])
    X = nr.words_to_trs(onsets, feats, n_tr, tr)
    assert X.shape == (n_tr, 1)
    t_peak1 = (X[:20, 0].argmax() + 0.5) * tr
    assert 13 < t_peak1 < 17
    # second pulse is twice as large
    assert abs(X[25:, 0].max() / X[:20, 0].max() - 2) < 0.05
    # lag shifts the response later
    X2 = nr.words_to_trs(onsets, feats, n_tr, tr, lag_s=3.0)
    assert X2[:20, 0].argmax() > X[:20, 0].argmax()


def test_words_to_trs_drops_words_after_scan_end():
    X = nr.words_to_trs(np.array([5.0, 500.0]), np.array([1.0, 1.0]), 20, 1.5)
    assert X.shape == (20, 1) and X.sum() > 0


def test_split_half_isc_detects_shared_signal():
    rng = np.random.default_rng(0)
    n_subj, n_tr, n_parc = 20, 200, 5
    shared = rng.standard_normal((n_tr, n_parc))
    weight = np.array([0.0, 0.3, 0.6, 1.0, 2.0])         # parcel 0 has no shared signal
    bold = shared[None] * weight[None, None] + rng.standard_normal((n_subj, n_tr, n_parc))
    isc = nr.split_half_isc(bold, n_splits=20)
    assert isc[0] < 0.2 and isc[4] > 0.9 and np.all(np.diff(isc) > 0)
    loo = nr.leave_one_out_isc(bold)
    assert np.all(np.diff(loo) > 0)


def test_network_label():
    assert nr.network_of("7Networks_LH_Default_PFC_1") == "Default"
