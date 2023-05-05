import numpy as np

from streamchange.data import simulate
from streamchange.amoc import CUSUM, CUSUM0, MaxCUSUM, SumCUSUM

estimator_classes = [CUSUM, CUSUM0, MaxCUSUM, SumCUSUM]


def test_cusum_nochange():
    x = simulate(seed=145).to_numpy()
    n = x.shape[0]
    for estimator_class in estimator_classes:
        estimator = estimator_class(0.0)
        estimator.fit(x)
        assert estimator.score > 0.0
        assert estimator.change_detected
        assert estimator.changepoint >= 1 and estimator.changepoint <= n - 1


def test_cusum_bigchange():
    seg_len = 50
    x = simulate(means=[30, 0], seg_lens=[seg_len], seed=145).to_numpy()
    for estimator_class in estimator_classes:
        estimator = estimator_class()
        estimator.fit(x)
        assert estimator.score > 0.0
        assert estimator.change_detected
        assert estimator.changepoint == seg_len


def test_cusum_candidates():
    x = simulate(seg_lens=[50], seed=145).to_numpy()
    n = x.shape[0]
    for estimator_class in estimator_classes:
        estimator = estimator_class(0.0)
        for minsl in [1, 2, 5]:
            candidate_cpts = np.arange(minsl, n - minsl + 1)
            estimator.fit(x, candidate_cpts)
            assert estimator.changepoint <= n - minsl
            assert estimator.changepoint >= minsl

        x = simulate(means=[0, 30], seg_lens=[5]).to_numpy()
        minsl = 6
        candidate_cpts = np.arange(minsl, x.shape[0] - minsl + 1)
        estimator = estimator_class(0.0)
        estimator.fit(x, candidate_cpts)
        assert estimator.changepoint is None
        assert not estimator.change_detected
        assert estimator.score < 0.0


def test_cusum_nan():
    for estimator_class in estimator_classes:
        estimator = estimator_class(0.0)
        inputs = [
            np.array([1, 1, 1, np.nan, 40, 40]).reshape(-1, 1),  # Nan
            # np.array([1, 1, 1, np.inf, np.inf]),
        ]
        for x in inputs:
            estimator.fit(x)
            assert not estimator.change_detected
            assert np.isnan(estimator.score)


# TODO: Add tests for ZeroPrechangeCUSUM and multivariate CUSUMs.
