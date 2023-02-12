from river.stats import Mean
import pandas as pd
import numpy as np

from streamchange.segment_stats import Buffer
from streamchange.data import simulate


def test_buffer_mean():
    n = 100
    series = simulate(seg_lens=[n], p=1)[0]
    stat = Buffer(Mean(), 20)
