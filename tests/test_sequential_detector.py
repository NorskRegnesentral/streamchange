# TODO
import numpy as np

from streamchange.data import simulate
from streamchange.sequential import (
    LordenPollakScore,
    CUSUM0Score,
    AggregatedScore,
    SequentialChangeDetector,
)

SCORE_CLASSES = [LordenPollakScore, CUSUM0Score]


def test_nochange():
    x = simulate(seed=145)[0]
    x_multivar = simulate(seed=145, p=5)

    for score_class in SCORE_CLASSES:
        score = score_class()
        score.fit(x)
        assert np.all(score.values_ >= 0.0)

        penalty = 100000
        penalised_score = score_class().penalise(penalty)
        penalised_score.fit(x)
        assert np.all(penalised_score.values_ < 0.0)

        base_score = score_class()
        score = AggregatedScore(base_score, aggregator=np.sum).penalise(penalty)
        detector = SequentialChangeDetector(
            score, reset_on_change=True, restart_delay=50
        )
        detector.fit(x_multivar)
        assert len(detector.alarms_) == 0
        assert np.all(detector.penalised_scores_ < 0.0)


def test_change():
    seg_len = 50
    x = simulate(means=[0, 30], seg_lens=[seg_len], seed=145)[0]
    x_multivar = simulate(means=[0, 30], seg_lens=[seg_len], seed=145, p=5)

    for score_class in SCORE_CLASSES:
        score = score_class()
        score.fit(x)
        assert np.all(score.values_ >= 0.0)

        penalty = 1.0
        penalised_score = score_class().penalise(penalty)
        penalised_score.fit(x)
        assert np.any(penalised_score.values_ >= 0.0)

        base_score = score_class()
        score = AggregatedScore(base_score, aggregator=np.sum).penalise(penalty)
        detector = SequentialChangeDetector(
            score, reset_on_change=True, restart_delay=50
        )
        detector.fit(x_multivar)
        assert len(detector.alarms_) > 0
        assert np.any(detector.penalised_scores_ >= 0.0)
