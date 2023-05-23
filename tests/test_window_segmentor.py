import pytest
from river.stream import iter_pandas
import numpy as np

from streamchange.amoc import CUSUM, CUSUM0, WindowSegmentor
from streamchange.penalties import BIC
from streamchange.data import simulate


def test_sane_accuracy():
    seg_len = 100
    df = simulate([0, 30], [seg_len], p=1, seed=2)
    estimator = CUSUM(penalty=BIC(scale=10))
    detector = WindowSegmentor(estimator, 4, 100)
    cpts = []
    for t, (x, _) in enumerate(iter_pandas(df)):
        detector.update(x)
        if detector.change_detected:
            cpts.append((t, detector.changepoints))
    assert len(cpts) == 1
    assert cpts[0][0] == seg_len
    assert cpts[0][1][0] == 1


def test_varying_threshold():
    seg_len = 100
    df = simulate([0, 10, 0], [seg_len], p=1, seed=5)
    penalties = [0.001, 0.1, 1, 2, 3, 4, 5, 10, 10000]
    for penalty in penalties:
        try:
            estimator = CUSUM(penalty=penalty)
            detector = WindowSegmentor(estimator, 2, 100)
            detector.fit(df)
        except Exception as exc:
            assert False, f"'detector.update()' raised exception {exc}"


def test_window_sizes():
    seg_len = 30
    df = simulate([0, 10, 0, 10, 0], [seg_len], p=1, seed=5)
    min_windows = [2, 10, 100]
    max_windows = [2, 50, 500]
    for min_window, max_window in zip(min_windows, max_windows):
        try:
            estimator = CUSUM()
            detector = WindowSegmentor(estimator, min_window, max_window)
            detector.fit(df)
        except Exception as exc:
            assert False, f"'detector.update()' raised exception {exc}"

    with pytest.raises(Exception):
        WindowSegmentor(CUSUM(), 10, 9)
    with pytest.raises(Exception):
        WindowSegmentor(CUSUM(), 1, 10)

    seg_len = 20
    df = simulate([0, 10, 0, 20, 0, 1, 0, 3, 5, 0, 4], [seg_len], p=1, seed=34)
    estimator = CUSUM(penalty=10)
    detector = WindowSegmentor(estimator, 4, 100)
    most_recent_cpt = 0
    for t, (x, _) in enumerate(iter_pandas(df)):
        detector.update(x)
        assert len(detector.window) <= detector.max_window
        assert len(detector.window) <= most_recent_cpt + 1
        if detector.change_detected:
            most_recent_cpt = detector.changepoints[-1]
        else:
            most_recent_cpt += 1


def test_minsl():
    seg_len = 50
    df = simulate([0, 10, 0], [seg_len], p=1, seed=5)
    minsls = [1, 2, 10]
    for minsl in minsls:
        detector = WindowSegmentor(CUSUM(), minsl=minsl)
        detector.fit(df)
        cpts = np.array(detector.changepoints_)
        assert np.all(np.diff(cpts) >= minsl)

    with pytest.raises(Exception):
        WindowSegmentor(CUSUM(), 2, 10, 6)
    with pytest.raises(Exception):
        WindowSegmentor(CUSUM0(), 2, 10, 11)


def test_candidates():
    seg_len = 50
    df = simulate([0, 10, 0], [seg_len], p=1, seed=5)
    try:
        detector = WindowSegmentor(CUSUM(), candidate_type="linear", candidate_step=2)
        detector.fit(df)
        detector = WindowSegmentor(CUSUM(), candidate_type="linear", candidate_step=5)
        detector.fit(df)
        detector = WindowSegmentor(CUSUM(), candidate_type="geom", candidate_step=1.1)
        detector.fit(df)
        detector = WindowSegmentor(CUSUM(), candidate_type="geom", candidate_step=2.0)
        detector.fit(df)
    except Exception as exc:
        assert False, f"'detector.fit()' raised exception {exc}"
