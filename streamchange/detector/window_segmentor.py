import numpy as np

from .change_detector import ChangeDetector
from .utils import get_public_properties
from streamchange.amoc_test import AMOCTest


class WindowSegmentor(ChangeDetector):
    """
    Class for testing-based changepoint detection.

    Parameters
    ----------
    min_window
        The minimum size of the window to compute the changepoint test over.
    max_window
        The maximum size of the window to compute the changepoint test over.
        This governs how many historical samples are retained in memory.
    """

    def __init__(
        self,
        test: AMOCTest,
        min_window: int = 2,
        max_window: int = np.inf,
    ):
        assert min_window >= 2
        assert max_window > min_window

        self.test = test
        self.max_window = max_window
        self.min_window = min_window
        self.window = DetectionWindow(max_window)
        self.reset()

    @property
    def change_detected(self):
        return len(self._changepoints) > 0

    @property
    def changepoints(self):
        """List of detected changepoints per iteration (call to update).

        Changepoints are stored as their negative index within the current window.
        This makes it easy to extract changepoints also outside this class,
        where the relevant temporal frame of reference is.
        """
        return self._changepoints

    def reset(self) -> "WindowSegmentor":
        self.window.reset()
        self._changepoints = []
        return self

    def _detect_changes(self):
        self._changepoints = []
        window = self.window.get()
        n = len(self.window)
        start = 0
        end = max(n, self.min_window)
        while end <= n:
            self.test.detect(window[start:end])
            if self.test.change_detected:
                self._changepoints.append(self.test.changepoint)
                start = (self.test.changepoint + n) + 1
                end = start + self.min_window
            else:
                end += 1

    def _detect_changes_generator(self):
        self._changepoints = []
        self.window.iterate()
        while self.window.keep_iterating:
            subset = self.window.next(self.test)
            self.test.detect(subset)
            if self.test.change_detected:
                self._changepoints.append(self.test.changepoint)
                self.window.popleft(self.test.changepoint + 1)

    def update(self, x):
        self.window.append(x)
        self._detect_changes()
        return self


class DetectionWindow:
    def __init__(self, max_length=np.inf):
        self.max_length = max_length
        self.reset()

    def reset(self) -> "DetectionWindow":
        self._w = None
        self.columns = None
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

        n = len(self.window)
        start = 0
        end = max(n, self.min_window)
        while end <= n:
            self.test.detect(window[start:end])
            if self.test.change_detected:
                self._changepoints.append(self.test.changepoint)
                start = (self.test.changepoint + n) + 1
                end = start + self.min_window
            else:
                end += 1

    def __len__(self):
        return 0 if self._w is None else self._w.shape[0]

    def iterate(self):
        self.start = 0
        self.end = max(len(self), self.min_window) - 1
        self.keep_iterating = self.end < len(self)

    def next(self, test: AMOCTest):
        if test.change_detected:
            n = len(self)
            start = (self.test.changepoint + n) + 1
            return self._w[start : start + self.min_window]
        else:
            self.end += 1
        return self._w[self.start : self.end]


# TODO:
# Implement classical sequential tests (entirely recursive).
# Possibility of several window mechanics:
#     Reset to min_window.,
#     Reset to t.
# Possibility to adjust candidate change-points (minimum and maximum seglen).
