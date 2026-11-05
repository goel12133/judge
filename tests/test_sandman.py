import numpy as np
import pytest

from judge import ceiling, sandman

pytestmark = pytest.mark.skipif(not sandman.DEFAULT_ROOT.exists(), reason="SuspensePerception repo not cloned")


@pytest.fixture(scope="module")
def s():
    return sandman.load()


def test_shape_and_range(s):
    assert s.ratings.shape == (23, 65)
    assert s.ratings.values.min() >= 1 and s.ratings.values.max() <= 10
    assert len(s.segments_de) == 65 and len(s.segments_en) == 65
    assert all(s.segments_en)


def test_published_means_match(s):
    # first values of the aggregate curve published in the repo's human_ratings.md
    m = s.mean_rating()
    assert np.isclose(m[0], 5.565217391, atol=1e-6)
    assert np.isclose(m[5], 6.826086957, atol=1e-6)
    assert np.isclose(m[49], 4.913043478, atol=1e-6)


def test_word_counts_match_paper(s):
    # Lehne: 6,859 words, M = 105.5 per segment; our paragraph split of the German is close
    n = sum(len(d.split()) for d in s.segments_de)
    assert 6500 < n < 7500


def test_ceiling_is_high(s):
    c = ceiling.split_half_ceiling(s.long, "reader", "segment", "y", n_splits=100)
    assert c["median"] > 0.8
