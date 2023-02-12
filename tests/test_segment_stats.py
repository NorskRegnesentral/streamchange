from river.stats import Mean
import pandas as pd
import numpy as np
import pytest

from streamchange.segment_stats import Buffer, StatUnion
from streamchange.data import simulate


def test_buffer_mean():
    n = 100
    series = simulate(seg_lens=[n], seed=34)[0]
    stat = Buffer(Mean(), 20)
    for _, x in series.items():
        stat.update(x)

    assert stat.get() == stat.get(-1)
    assert stat.get() >= -1 and stat.get() <= 1
    with pytest.raises(ValueError):
        stat.get(-stat.max_history - 1)

    stat.reset()
    assert abs(stat.get()) < 1e-8
    stat.update(1.0)
    with pytest.raises(IndexError):
        stat.get(-2)


def test_stat_union():
    n = 100
    series = simulate(seg_lens=[n], seed=34)[0]
    stat = StatUnion({"mean": Buffer(Mean())}, 20)
    for _, x in series.items():
        stat.update(x)

    assert stat.get() == stat.get(-1)
    with pytest.raises(ValueError):
        stat.get(-stat.max_history - 1)

    stat.reset()
    stat.update(1.0)
    with pytest.raises(IndexError):
        stat.get(-2)
