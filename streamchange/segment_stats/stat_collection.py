import numbers
import numpy as np
import river
from collections import deque

from .base import SegmentStat

class StatCollection(SegmentStat):
    def __init__(self, stats: dict):
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


    # def restart(self, X: np.ndarray):
    #     pass

    # def get_restart(self, cpts, X: np.ndarray) -> list:
    #     return {name: stat.get_restart(cpts, X) for name, stat in self.items()}
