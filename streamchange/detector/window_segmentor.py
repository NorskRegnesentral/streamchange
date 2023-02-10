import numpy as np
from numbers import Number
from typing import Union

from .change_detector import ChangeDetector
from streamchange.amoc_test import AMOCTest


class WindowSegmentor(ChangeDetector):
    """
    Class for testing-based changepoint detection.

    Parameters
    ----------
    test
        The single changepoint test.
    min_window
        The minimum length of the window to test for changepoints in.
    max_window
        The maximum length of the window to test for changepoints in.
    with_jumpback
        Upon detection of a changepoint, whether to jump back to a minimum
        length window starting right after the changepoints.
    """

    def __init__(
        self,
        test: AMOCTest,
        min_window: int = 2,
        max_window: int = np.inf,
        with_jumpback: bool = True,
    ):
        self.test = test
        self.min_window = min_window
        self.max_window = max_window
        self.window = DetectionWindow(max_window)
        self.with_jumpback = with_jumpback
        self.reset()

    def reset(self) -> "WindowSegmentor":
        self._changepoints = []
        self.test.reset()
        self.window.reset()
        return self

    def update(self, x):
        if self.change_detected:
            self.window.popleft(self.changepoints[-1] + 1)
        self.window.append(x)
        self._changepoints = []

        w = self.window.get()
        start = 0
        end = max(len(self.window), self.min_window)
        while end <= len(self.window):
            self.test.detect(w[start:end])
            if self.test.change_detected:
                cpt = self.test.changepoint
                self._changepoints.append(cpt)
                if self.with_jumpback:
                    start = cpt + 1
                    end = cpt + self.min_window - 1
            end += 1
        return self


class DetectionWindow:
    def __init__(self, max_length=np.inf):
        """
        Parameters
        ----------
        max_length:
            The maximum size of the window to compute the changepoint test over.
            This governs how many historical samples are retained in memory.
        """
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
