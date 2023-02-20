import numpy as np

from .change_detector import ChangeDetector
from streamchange.amoc_test import AMOCTest
from streamchange.base import NumpyDeque


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
        self.window = NumpyDeque(max_window)
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

        start = 0
        end = max(len(self.window), self.min_window)
        while end <= len(self.window):
            self.test.detect(self.window.values[start:end])
            if self.test.change_detected:
                cpt = self.test.changepoint
                self._changepoints.append(cpt)
                if self.with_jumpback:
                    start = cpt + 1
                    end = cpt + self.min_window - 1
            end += 1
        return self
