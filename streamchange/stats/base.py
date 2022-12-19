import abc
import numbers
import numpy as np

class SegmentStat(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> "SegmentStat":
        pass

    @abc.abstractmethod
    def revert(self, X: np.ndarray) -> "SegmentStat":
        pass

    @abc.abstractmethod
    def restart(self, X: np.ndarray) -> "SegmentStat":
        pass
