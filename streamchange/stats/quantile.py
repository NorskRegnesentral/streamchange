import river

from .base import SegmentStat


class Quantile(river.stats.Quantile, SegmentStat):
    def reset(self):
        self._quantile = _rust_stats.RsQuantile(q)
        self._is_updated = False

    def revert():
        pass

    def restart():
        pass
