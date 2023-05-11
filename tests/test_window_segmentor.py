from river.stream import iter_pandas

from streamchange.amoc import CUSUM, WindowSegmentor
from streamchange.penalties import BIC
from streamchange.data import simulate


def test_accuracy():
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
            for t, (x, _) in enumerate(iter_pandas(df)):
                detector.update(x)
        except Exception as exc:
            assert False, f"'detector.update()' raised exception {exc}"


def test_detection_window_size():
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
