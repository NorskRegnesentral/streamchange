from .base import BaseSegmentStat

import numbers
import numpy as np
import river.stats as river_stats
from collections import deque


class StatBuffer(BaseSegmentStat):
    def __init__(self, stat: river_stats.base.Univariate, max_history=np.inf):
        super().__init__(max_history)
        self.stat = stat
        self.reset()

    def reset(self) -> "BaseSegmentStat":
        self.stat = self.stat.clone(include_attributes=False)
        self._buffer = (
            deque() if np.isinf(self.max_history) else deque(maxlen=self.max_history)
        )
        return self

    def get(self, i=0) -> numbers.Number:
        self.check_get(i)
        if i == 0:
            return self.stat.get()
        else:
            return list(self._buffer)[i]

    def update(self, x: numbers.Number) -> "BaseSegmentStat":
        self.stat.update(x)
        self._buffer.appendleft(self.stat.get())
        return self

    def __len__(self):
        return len(self._buffer)
