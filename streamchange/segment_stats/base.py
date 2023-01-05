import abc
import numbers
import numpy as np


class SegmentStat(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> "SegmentStat":
        return self

    @abc.abstractmethod
    def get(self, i: int = -1) -> numbers.Number:
        pass

    @abc.abstractmethod
    def update(self, x: numbers.Number) -> "SegmentStat":
        return self

    def update_many(self, x: np.ndarray) -> "SegmentStat":
        for value in np.nditer(x.squeeze()):
            self.update(value)
        return self

    def restart(self, x: np.ndarray) -> "SegmentStat":
        return self.reset().update_many(x)

    def get_restart(self, cpts: list, x: np.ndarray) -> list:
        cpts = cpts + [x.size-1]
        stats = []
        for curr_cpt, next_cpt in zip(cpts[:-1], cpts[1:]):
            stats.append(self.get(curr_cpt))
            self.restart(x[curr_cpt + 1 : next_cpt + 1])
        return stats
