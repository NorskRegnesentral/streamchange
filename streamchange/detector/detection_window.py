import abc
import numpy as np
from numbers import Number
from typing import Union

from streamchange.amoc_test import AMOCTest


class DetectionWindow:
    def __init__(self, min_length=2, max_length=np.inf):
        """
        min_window
            The minimum size of the window to compute the changepoint test over.
        max_window
            The maximum size of the window to compute the changepoint test over.
            This governs how many historical samples are retained in memory.
        """
        assert min_length >= 2
        assert max_length > min_length

        self.min_length = min_length
        self.max_length = max_length
        self.reset()

    def reset(self) -> "DetectionWindow":
        self.columns = None
        self.p = None
        self._w = None
        return self

    def get(self) -> np.ndarray:
        return self._w

    def popleft(self, n: int = 1) -> np.ndarray:
        self._w = self._w[n:]
        return self._w[:n]

    def _init_window(self, x):
        if isinstance(x, Number):
            self.columns = None
            self.p = 1
            self._w = np.empty((0, 1))
        else:
            self.columns = list(x.keys())
            self.p = len(self.columns)
            self._w = np.empty((0, self.p))

    def append(self, x: Union[Number, dict]):
        if self._w is None:
            self._init_window(x)

        next_row = [x] if isinstance(x, Number) else [x[key] for key in self.columns]
        self._w = np.concatenate((self._w, np.array([next_row])))
        if len(self) > self.max_length:
            self.popleft()

    def __len__(self) -> int:
        return 0 if self._w is None else self._w.shape[0]

    def init_iter(self) -> "DetectionWindow":
        self.end = max(len(self), self.min_length)
        return self

    def stop_iter(self) -> bool:
        return self.end > len(self)

    def next(self) -> np.ndarray:
        subset = self._w[: self.end]
        self.end += 1
        return subset

    @abc.abstractmethod
    def detection_reaction(self, test: AMOCTest) -> "DetectionWindow":
        return self


class SlidingWindow(DetectionWindow):
    def detection_reaction(self, test):
        self.popleft(test.changepoint + 1)
        return self


class JumpbackWindow(DetectionWindow):
    def detection_reaction(self, test):
        self.popleft(test.changepoint + 1)
        self.end = self.min_length
        return self


class ResetWindow(DetectionWindow):
    def detection_reaction(self, test):
        self.reset()
        return self
