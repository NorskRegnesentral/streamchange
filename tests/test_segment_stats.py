from river.stats import Mean
import pandas as pd
import numpy as np
import pytest

from streamchange.segment_stats import StatBuffer, StatUnion
from streamchange.data import simulate


def test_StatBuffer_mean():
    n = 100
    series = simulate(seg_lens=[n], seed=34)[0]
    stat = StatBuffer(Mean(), 20)
    for _, x in series.items():
        stat.update(x)

    assert stat.get() == stat.get(0)
    assert stat.get() >= -1 and stat.get() <= 1
    with pytest.raises(IndexError):
        stat.get(stat.max_history)

    stat.reset()
    assert abs(stat.get()) < 1e-8
    stat.update(1.0)
    with pytest.raises(IndexError):
        stat.get(1)


def test_stat_union():
    n = 100
    series = simulate(seg_lens=[n], seed=34)[0]
    stat = StatUnion({"mean": StatBuffer(Mean())}, 20)
    for _, x in series.items():
        stat.update(x)

    with pytest.raises(IndexError):
        stat.get(stat.max_history)

    stat.reset()
    stat.update(1.0)
    with pytest.raises(IndexError):
        stat.get(1)
