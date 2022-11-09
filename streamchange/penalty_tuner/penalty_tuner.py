from streamchange.detector.segmentor import Segmentor, window_start
from streamchange.detector.segmentor_collection import SegmentorCollection

import pandas as pd
import numpy as np
from numba import njit


@njit
def generate_intervals(
    data_size: int,
    min_size_window: int,
    max_size_window: int,
    sampling_probability: float = 1.0,
):
    starts = []
    ends = []
    for end in range(min_size_window, data_size):
        min_start = window_start(end, max_size_window)
        max_start = end - min_size_window + 1
        for start in range(min_start, max_start):
            if np.random.uniform(0.0, 1.0) <= sampling_probability:
                starts.append(start)
                ends.append(end)
    return starts, ends


class PenaltyTuner:
    """
    Class for tuning the penalty of a segmentor.
    Can be used for tuning and storing a penalty for later use, or to
    set the penalty in a segmentor directly.
    """

    def __init__(self, max_cpts: int = 1000, sampling_probability: float = 1.0):
        self.max_cpts = max_cpts
        self.sampling_probability = sampling_probability

    def __call__(self, segmentor, data):
        self.tune(segmentor, data)

    def find_penalties(self, segmentor: Segmentor, data: pd.Series) -> np.ndarray:
        assert not pd.isnull(data).any()
        assert data.size >= self.max_cpts

        starts, ends = generate_intervals(
            data.size,
            segmentor.min_size_window,
            segmentor.max_size_window,
            self.sampling_probability,
        )
        tests, cpts = segmentor.tests_for(starts, ends, data.to_numpy())
        tests = np.array(tests)  # For quicker indexing below.
        starts = np.array(starts)  # For quicker indexing below.
        ends = np.array(ends)  # For quicker indexing below.

        # Find threshold c per number of change-points.
        penalties = np.zeros(self.max_cpts)
        i = 0
        while (i < self.max_cpts) & np.any(tests > 0.0):
            argmax = tests.argmax()
            penalties[i] = tests[argmax]
            tau = cpts[argmax]
            tests[(tau >= starts) & (tau < ends)] = 0.0
            i += 1

        self.penalties = penalties  # Store to be able to evaluate and plot.
        return penalties

    def select_penalty(self, penalties):
        pass

    def tune_segmentor(self, segmentor: Segmentor, data: pd.Series):
        penalties = self.find_penalties(segmentor, data.dropna())
        segmentor.penalty = self.select_penalty(penalties)

    def tune_segmentor_collection(
        self, segmentor: SegmentorCollection, data: pd.DataFrame
    ):
        for name in segmentor.keys():
            self.tune_segmentor(segmentor[name], data[name])

    def tune(self, segmentor, data):
        if isinstance(segmentor, Segmentor):
            self.tune_segmentor(segmentor, data)
        elif isinstance(segmentor, SegmentorCollection):
            self.tune_segmentor_collection(segmentor, data)
        else:
            raise ValueError(
                "tune(segmentor, data) requires segmentor to be an instance of"
                " Segmentor or SegmentorCollection."
            )

    def show(self):
        pass
