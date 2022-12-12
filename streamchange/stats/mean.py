import river

from .base import SegmentStat

class Mean(river.stats.Mean, SegmentStat):

    def reset(self):
        self.n = 0
        self._mean = 0.0
        return self 

    def restart(self, X):
        self.reset()
        self.update_many(X)
        return self
