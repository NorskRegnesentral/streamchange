import numbers
import numpy as np
import river
from collections import deque
from typing import Dict

from .base import BaseSegmentStat


class StatUnion(BaseSegmentStat):
    def __init__(self, stats: Dict[str, BaseSegmentStat], max_history=np.inf):
        super().__init__(max_history)
        for stat in stats.values():
            stat.max_history = max_history
        self.stats = stats

    def __getitem__(self, key):
        return self.stats[key]

    def __len__(self):
        return len(self.stats)

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

    def get(self, i=0) -> Dict[str, float]:
        self.check_get(i)
        return {name: stat.get(i) for name, stat in self.items()}

    def update(self, x):
        for stat in self.values():
            stat.update(x)
        return self
