from streamchange.detector.change_detector import ChangeDetector

import abc
import pandas as pd
import numpy as np
from numba import njit


@njit
def generate_intervals(
    data_size: int,
    min_window: int,
    max_window: int,
    sampling_probability: float = 1.0,
):
    starts = []
    ends = []
    for end in range(min_window, data_size):
        min_start = max(0, end - max_window)
        max_start = end - min_window + 1
        for start in range(min_start, max_start):
            if np.random.uniform(0.0, 1.0) <= sampling_probability:
                starts.append(start)
                ends.append(end)
    return starts, ends


class ThresholdTuner:
    """
    Class for tuning the threshold of a segmentor.
    Can be used for tuning and storing a threshold for later use, or to
    set the threshold in a detector directly.
    """

    def __init__(self, max_cpts: int = 1000, sampling_probability: float = 1.0):
        self.max_cpts = max_cpts
        self.sampling_probability = sampling_probability

    def __call__(self, detector, data):
        self.tune(detector, data)

    def _detect_and_locate_in(self, starts: list, ends: list, x: np.ndarray):
        """
        Outputs the optimal test statistic and maximising argument (changepoint)
        of each interval given by start[i]:(end[i]) in x.
        The purpose of this method is to facilitate computationally efficient
        threshold tuning for specific segmentors. I.e., it should be overwritten
        by a subclass.
        See UnivariateCUSUM for example.
        """
        tests, cpts = zip(*[self.test(x[s:e]) for s, e in zip(starts, ends)])
        tests = list(tests)
        cpts = list(np.array(starts) + np.array(cpts))
        return tests, cpts

    def find_penalties(self, detector: ChangeDetector, data: pd.Series) -> np.ndarray:
        assert not pd.isnull(data).any()
        assert data.size >= self.max_cpts

        starts, ends = generate_intervals(
            data.size,
            detector.min_window,
            detector.max_window,
            self.sampling_probability,
        )
        tests, cpts = detector._tests_for(starts, ends, data.to_numpy())
        tests = np.array(tests)  # For quicker indexing below.
        starts = np.array(starts)  # For quicker indexing below.
        ends = np.array(ends)  # For quicker indexing below.

        # Find threshold c per number of change-points.
        penalties = np.zeros(self.max_cpts)
        i = 0
        while (i < self.max_cpts) & np.any(tests > 0.0):
            argmax = tests.argmax()
            penalties[i] = tests[argmax]
            max_cpt = cpts[argmax]
            cpt_in_interval = (max_cpt >= starts) & (max_cpt < ends)
            tests[cpt_in_interval] = 0.0
            i += 1

        self.penalties = penalties  # Store to be able to evaluate and plot.
        return penalties

    @abc.abstractmethod
    def select_threshold(self, penalties):
        """
        Threshold selection procedure.
        """

    def tune(self, detector: ChangeDetector, data: pd.Series):
        self._detector = detector
        self._data = data.dropna()
        penalties = self._find_penalties()
        detector.threshold = self._select_threshold(penalties)

    def show(self):
        pass
