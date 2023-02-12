import numbers
import numpy as np
import river
from collections import deque

from .base import SegmentStat


class StatUnion(SegmentStat):
    def __init__(self, stats: dict, max_history = -np.inf):
        super().__init__(max_history)
        for stat in stats.values():
            stat.max_history = max_history
        self.stats = stats

    def items(self):
        return self.stats.items()

    def values(self):
        return self.stats.values()

    def keys(self):
        return self.stats.keys()

    def reset(self):
        for stat in self.values():
            stat.reset()
        return self

    def get(self, i=-1) -> dict:
        return {name: stat.get(i) for name, stat in self.items()}

    def update(self, x):
        for stat in self.values():
            stat.update(x)
        return self
