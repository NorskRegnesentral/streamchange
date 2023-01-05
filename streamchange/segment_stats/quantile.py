import river
from river.stats import _rust_stats
from collections import deque

from .base import SegmentStat


class Quantile(river.stats.Quantile, SegmentStat):
    def __init__(self, q: float = 0.5, maxlen_revert: int = 1):
        super().__init__(q)
        self.maxlen_revert = maxlen_revert
        self._history = deque(maxlen=maxlen_revert)

    def reset(self):
        self._quantile = _rust_stats.RsQuantile(self.q)
        self._is_updated = False
        self._history = deque(maxlen=self.maxlen_revert)
        return self

    def update(self, x):
        super().update(x)
        self._history.append(self._quantile.get())
        return self

    def revert(self, n):

        return self
