import numbers
import abc
import numpy as np


class ChangeDetector:
    """
    Base class for testing-based changepoint detection.

    Parameters
    ----------
    threshold
        Dete detection threshold.
    min_window
        The minimum size of the window to compute the changepoint test over.
    max_window
        The maximum size of the window to compute the changepoint test over.
        This governs how many historical samples are retained in memory.
    """

    def __init__(
        self,
        threshold: numbers.Number = 2.0 * np.log(10000),
        min_window: int = 2,
        max_window: int = np.inf,
    ):

        assert threshold >= 0
        assert min_window >= 2
        assert max_window > min_window

        self.threshold = threshold
        self.min_window = min_window
        self.max_window = max_window
        self._reset()

    def _reset(self):
        self._change_detected = False
        self._window = np.array([], dtype="float64")

    @abc.abstractmethod
    def test(self, x: np.ndarray) -> np.ndarray:
        """Compute the changepoint test

        Parameters
        ----------
        x
            Input values.

        Returns
        -------
        cpt_test
            The value of the changepoint test statistic.
        cpt
            The changepoint location within x.
        """

    def _update_window(self, value: numbers.Number):
        if self._change_detected:
            start = self._window.size - self.cpts[-1] + 1
        else:
            start = max(0, self._window.size - self.max_window + 1)
        self._window = np.concatenate((self._window[start:], np.array([value])))

    def _detect_changes(self):
        self.cpts = []
        start = 0
        end = max(self._window.size, self.min_window)
        while end <= self._window.size:
            cpt_test, cpt = self.test(self._window[start:end])
            if cpt_test > self.threshold:
                self.cpts.append(self._window.size - cpt - 1)
                start = cpt + 1
                end = start + self.min_window
            else:
                end += 1
        self._change_detected = True if len(self.cpts) > 0 else False

    def update(self, value: numbers.Number):
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            Input value.

        Returns
        -------
        self
        """
        self._update_window(value)
        self._detect_changes()
        return self

    def _tests_for(self, starts: list, ends: list, x: np.ndarray):
        """
        Outputs the optimal test statistic and maximising argument (changepoint)
        of each interval given by start[i]:(end[i]) in x.
        The purpose of this method is to facilitate computationally efficient
        threshold tuning for specific segmentors. I.e., it should be overwritten
        by a subclass.
        See UnivariateCUSUM for example.
        """
        tests, cpts = zip(*[self.test(x[s:e]) for s, e in zip(starts, ends)])
        return list(tests), list(np.array(starts) + np.array(cpts))
