import abc
import numbers
import numpy as np


class SegmentStat(abc.ABC):
    def __init__(self, max_history=np.inf):
        assert max_history >= 1
        self.max_history = max_history

    @abc.abstractmethod
    def reset(self) -> "SegmentStat":
        return self

    def check_get(self, i):
        if i >= 0:
            raise IndexError("i must be negative.")
        elif i < -self.max_history:
            raise IndexError(
                f"Cannot get value of SegmentStat beyond {self.max_history} steps back"
                f" (i={i})."
            )

    @abc.abstractmethod
    def get(self, i: int = -1) -> numbers.Number:
        """Get value of statistic -i steps ago."""
        self.check_get(i)

    @abc.abstractmethod
    def update(self, x: numbers.Number) -> "SegmentStat":
        return self

    def update_many(self, x: np.ndarray) -> "SegmentStat":
        for value in np.nditer(x.squeeze()):
            self.update(value)
        return self
