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

    # assert stat[] == stat[-1]
    assert stat[-1] >= -1 and stat[-1] <= 1
    with pytest.raises(IndexError):
        stat[-stat.max_history - 1]

    stat.reset()
    assert abs(stat[-1]) < 1e-8
    stat.update(1.0)
    with pytest.raises(IndexError):
        stat[-2]


def test_stat_union():
    n = 100
    series = simulate(seg_lens=[n], seed=34)[0]
    stat = StatUnion({"mean": StatBuffer(Mean())}, 20)
    for _, x in series.items():
        stat.update(x)

    # assert stat[-1] == stat.get(-1)
    with pytest.raises(IndexError):
        stat[-stat.max_history - 1]

    stat.reset()
    stat.update(1.0)
    with pytest.raises(IndexError):
        stat[-2]
