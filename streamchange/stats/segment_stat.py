from .base import SegmentStat

import numbers
import numpy as np
import river.stats as river_stats
from collections import deque


class Buffer(SegmentStat):
    def __init__(self, stat: river_stats.base.Univariate, max_restart=1000):
        self.stat = stat
        self.max_restart = max_restart
        self.reset()

    def reset(self) -> "SegmentStat":
        self.stat = self.stat.clone(include_attributes=False)
        self._buffer = deque(maxlen=self.max_restart)
        return self

    def get(self, i: int = -1) -> numbers.Number:
        if i == -1:
            return self.stat.get()
        else:
            return list(self._buffer)[i]

    def update(self, x: numbers.Number) -> "SegmentStat":
        self.stat.update(x)
        self._buffer.append(self.stat.get())
        return self
