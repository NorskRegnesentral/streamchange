import abc
import numbers
import numpy as np


class BaseSegmentStat(abc.ABC):
    def __init__(self, max_history=np.inf):
        assert max_history >= 1
        self.max_history = max_history

    @abc.abstractmethod
    def reset(self) -> "BaseSegmentStat":
        return self

    def check_get(self, i):
        if i < 0:
            raise IndexError(f"i must be positive (i={i}).")
        elif i >= self.max_history:
            msg = f"Cannot get value of BaseSegmentStat beyond {self.max_history-1} steps back (i={i})."
            raise IndexError(msg)

    @abc.abstractmethod
    def get(self, i: int = 0) -> numbers.Number:
        """Get value of the statistic i steps ago.

        i = 0 means the current value.
        """
        self.check_get(i)

    @abc.abstractmethod
    def update(self, x: numbers.Number) -> "BaseSegmentStat":
        return self

    def update_many(self, x: np.ndarray) -> "BaseSegmentStat":
        for value in np.nditer(x.squeeze()):
            self.update(value)
        return self
