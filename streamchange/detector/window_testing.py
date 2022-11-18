import numpy as np

from .change_detector import ChangeDetector
from .utils import get_public_properties
from streamchange.amoc_test import AMOCTest


class WindowTesting(ChangeDetector):
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
        fetch_test_results: bool = False,
    ):
        assert min_window >= 2
        assert max_window > min_window

        self.test = test
        self.min_window = min_window
        self.max_window = max_window
        self.fetch_test_results = fetch_test_results
        self._reset()

    @property
    def change_detected(self):
        return len(self._changepoints) > 0

    @property
    def changepoints(self):
        return self._changepoints

    def _reset(self):
        self._reset_results()
        self._reset_window()

    def _reset_results(self):
        self._changepoints = []
        if self.fetch_test_results:
            self.test_results = []

    def _append_results(self):
        n = self._window.shape[0]
        self._changepoints.append((n - 1) - self.test.changepoint)
        if self.fetch_test_results:
            self.test_results.append(get_public_properties(self.test))

    def _reset_window(self):
        self._window = None

    def _init_window(self, x: np.ndarray):
        if len(x.shape) == 1:
            self._window = np.empty((0, 1))
        elif len(x.shape) == 2:
            self._window = np.empty((0, x.shape[1]))
        else:
            ValueError("x must be 1- or 2-dimensional.")

    def _update_window(self, x: np.ndarray):
        n = self._window.shape[0]
        if self.change_detected:
            new_start = (n - 1) - self._changepoints[-1] + 1
        else:
            new_start = max(0, n - self.max_window + 1)
        self._window = np.concatenate((self._window[new_start:], x))

    def _detect_changes(self):
        self._reset_results()
        n = self._window.shape[0]
        start = 0
        end = max(n, self.min_window)
        while end <= n:
            self.test.detect(self._window[start:end])
            if self.test.change_detected:
                # Changepoints are stored as their distance from the their
                # index of detection. This provides simple extraction of
                # changepoints also outside of this method, where there might be
                # some other temporal frame of reference.
                self._append_results()
                start = self.test.changepoint + 1
                end = start + self.min_window
            else:
                end += 1

    def update(self, x: np.ndarray):
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            One observation row-vector.

        Returns
        -------
        self
        """
        if self._window is None:
            self._init_window(x)

        self._update_window(x)
        self._detect_changes()
        return self
