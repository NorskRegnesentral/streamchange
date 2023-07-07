from streamchange.data import simulate
from streamchange.pelt import Pelt, L2Cost


def test_accuracy():
    seg_len = 100
    df = simulate([0, 30, 0], [seg_len], p=1, seed=2)
    detector = Pelt(L2Cost(), minsl=2, maxsl=10000)
    segments = detector.fit_predict(df)
    assert len(segments) == 3
    assert len(detector.changepoints_) == 2
    assert detector.changepoints_[0] == 2 * seg_len - 1
    assert detector.changepoints_[1] == seg_len - 1
    for segment in segments:
        assert segment["start"] >= 0
        assert segment["end"] <= df.size - 1
        assert segment["start"] <= segment["end"]
