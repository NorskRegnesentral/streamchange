import river
from river.stats import _rust_stats


from .base import SegmentStat


class Quantile(river.stats.Quantile, SegmentStat):
    def reset(self):
        self._quantile = _rust_stats.RsQuantile(self.q)
        self._is_updated = False
        return self

    def revert(self, x):
        return self

    def restart(self, X):
        self.reset()
        for x in X:
            self.update(x)
        return self
