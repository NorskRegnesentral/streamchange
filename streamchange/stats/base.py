import abc
import numbers
import numpy as np
import river
from collections import deque


class SegmentStat(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> "SegmentStat":
        pass

    @abc.abstractmethod
    def update(self, x: numbers.Number) -> "SegmentStat":
        pass

    @abc.abstractmethod
    def revert(self, X: np.ndarray) -> "SegmentStat":
        pass

    @abc.abstractmethod
    def restart(self, X: np.ndarray) -> "SegmentStat":
        self.reset()
        for x in X:
            self.update(x)
        return self


class Revertible:
    def __init__(self, stat: river.stats.Univariate, max_revert: int = 1):
        self.stat = stat
        self.max_revert = max_revert
        self._history = deque(maxlen=max_revert)

    def reset(self) -> "SegmentStat":
        self.stat = self.stat.__init__()
        pass

    def update(self, x: numbers.Number) -> "Revertible":
        self.stat.update(x)
        self._history.append(self.stat)
        return self

    @abc.abstractmethod
    def revert(self, n) -> "Revertible":
        pass

    @abc.abstractmethod
    def restart(self, X: np.ndarray) -> "Revertible":
        self.reset()
        for x in X:
            self.update(x)
        return self


class Dequified:
    def __init__(self, stat_class, maxlen=100, *args, **kwargs):
        self.stat_class = stat_class
        self._args = args
        self._kwargs = kwargs
        self.maxlen = maxlen
        self.reset()

    def reset(self) -> "Dequified":
        self.stat = self.stat_class(*self._args, **self._kwargs)
        self._history = deque(maxlen=self.maxlen)
        return self

    def update(self, x: numbers.Number) -> "Dequified":
        self.stat.update(x)
        self._history.append(self.stat.get())
        return self

    def get(self, index=-1) -> numbers.Number:
        return np.array(self._history)[index]

    def restart(self, X: np.ndarray) -> "Dequified":
        self.reset()
        for x in X:
            self.update(x)
        return self

    def get_restart(self, cpts, X: np.ndarray) -> list:
        stats = []
        n = X.size
        cpts = [(n - 1) - cpt for cpt in cpts + [0]]
        for curr_cpt, next_cpt in zip(cpts[:-1], cpts[1:]):
            stats.append(self.get(curr_cpt - n))
            self.restart(X[curr_cpt + 1 : next_cpt + 1])
        return stats
