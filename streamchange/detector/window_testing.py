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
        """List of detected changepoints per iteration (call to update).

        Changepoints are stored as their distance from the last observation.
        This makes it easy to extract changepoints also outside this class,
        where the relevant temporal frame of reference is.
        """
        return self._changepoints

    def _reset(self):
        self._reset_results()
        self._window = None
        self._variable_names = None

    def _init_variable_names(self, x: dict):
        self._variable_names = list(x.keys())

    def _reset_results(self):
        self._changepoints = []
        if self.fetch_test_results:
            self.test_results = []

    def _append_results(self):
        n = self._window.shape[0]
        self._changepoints.append((n - 1) - self.test.changepoint)
        if self.fetch_test_results:
            self.test_results.append(get_public_properties(self.test))

    def _init_window(self, x: dict):
        self._window = np.empty((0, len(x)))

    def _to_nprow(self, x: dict):
        p = len(self._variable_names)
        return np.array([x[name] for name in self._variable_names]).reshape(1, p)

    def _update_window(self, x: dict):
        n = self._window.shape[0]
        if self.change_detected:
            most_recent_cangepoint = (n - 1) - self._changepoints[-1]
            start = most_recent_cangepoint + 1
        else:
            start = max(0, n - self.max_window + 1)
        self._window = np.concatenate((self._window[start:], self._to_nprow(x)))

    def _detect_changes(self):
        self._reset_results()
        n = self._window.shape[0]
        start = 0
        end = max(n, self.min_window)
        while end <= n:
            self.test.detect(self._window[start:end])
            if self.test.change_detected:
                self._append_results()
                start = self.test.changepoint + 1
                end = start + self.min_window
            else:
                end += 1

    def update(self, x):
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

        if self._variable_names is None:
            self._init_variable_names(x)

        self._update_window(x)
        self._detect_changes()
        return self
