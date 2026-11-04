import numpy as np
import pytest

from judge import naturalstories as ns

pytestmark = pytest.mark.skipif(not ns.DEFAULT_ROOT.exists(),
                                reason="Natural Stories corpus not downloaded (data/raw)")


@pytest.fixture(scope="module")
def table():
    return ns.load()


def test_shape(table):
    assert table.words["item"].nunique() == 10
    assert len(table.words) > 10000
    assert table.rts["WorkerId"].nunique() > 150


def test_surprisal_alignment(table):
    w = table.words
    # coverage: almost every word has a surprisal after alignment
    assert w["surprisal"].isna().mean() < 0.01
    # known tokens from the first sentence of story 1 (GPT-3 davinci logprobs file)
    first = w[w["item"] == 1].set_index("zone")
    assert np.isclose(first.loc[2, "surprisal"], 0.7762714, atol=1e-5)   # "you"
    assert np.isclose(first.loc[5, "surprisal"], 10.098377, atol=1e-4)   # "journey"


def test_surprisal_predicts_rt(table):
    m = ns.mean_rt(table)
    r = m[["surprisal", "y"]].dropna().corr().iloc[0, 1]
    assert 0.08 < r < 0.4   # classic word-level effect, without spillover
