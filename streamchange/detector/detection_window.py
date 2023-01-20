import abc
import numpy as np

from streamchange.amoc_test import AMOCTest


class DetectionWindow:
    def __init__(self, min_length=2, max_length=np.inf):
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

    def append(self, x: dict):
        if self._w is None:
            self.columns = list(x.keys())
            self.p = len(self.columns)
            self._w = np.empty((0, self.p))

        next_row = np.array([[x[name] for name in self.columns]])
        self._w = np.concatenate((self._w, next_row))
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
    def detection_reaction(self, test: AMOCTest):
        self.popleft(test.changepoint + 1)
        return self


class JumpbackWindow(DetectionWindow):
    def detection_reaction(self, test: AMOCTest):
        self.popleft(test.changepoint + 1)
        self.end = self.min_length
        return self
