import numpy as np

from streamchange.data import simulate
from streamchange.amoc import UnivariateCUSUM


def test_cusum_nochange():
    x = simulate(seed=145).to_numpy()
    n = x.shape[0]
    test = UnivariateCUSUM(threshold=0.0, minsl=1)
    test.detect(x)
    assert test.score > 0.0
    assert test.change_detected
    assert test.changepoint >= -n and test.changepoint <= -2


def test_cusum_bigchange():
    seg_len = 50
    x = simulate(means=[0, 30], seg_lens=[seg_len], seed=145).to_numpy()
    test = UnivariateCUSUM(minsl=1).set_default_threshold(x.shape[0])
    test.detect(x)
    assert test.score > 0.0
    assert test.change_detected
    assert test.changepoint == -seg_len - 1


def test_minsl():
    x = simulate(seg_lens=[50], seed=145).to_numpy()
    n = x.shape[0]
    for minsl in [1, 2, 5]:
        test = UnivariateCUSUM(threshold=0.0, minsl=minsl)
        test.detect(x)
        assert test.changepoint >= -n - 1 + test.minsl
        assert test.changepoint <= -test.minsl - 1

    test = UnivariateCUSUM(threshold=0.0, minsl=6)
    test.detect(simulate(means=[0, 30], seg_lens=[5]))
    assert test.changepoint is None
    assert not test.change_detected
    assert test.score < 0.0


def test_cusum_nan():
    test = UnivariateCUSUM(threshold=0.0, minsl=1)
    inputs = [
        np.array([1, 1, 1, np.nan, 40, 40]),  # Nan
        np.array([1, 1, 1, np.inf, np.inf]),
    ]
    for x in inputs:
        test.detect(x)
        assert not test.change_detected
        assert np.isnan(test.score)
