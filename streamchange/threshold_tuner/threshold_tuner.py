from streamchange.detector.change_detector import ChangeDetector

# from streamchange.detector.segmentor_collection import SegmentorCollection

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
    set the threshold in a segmentor directly.
    """

    def __init__(self, max_cpts: int = 1000, sampling_probability: float = 1.0):
        self.max_cpts = max_cpts
        self.sampling_probability = sampling_probability

    def __call__(self, segmentor, data):
        self.tune(segmentor, data)

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
            tests[(starts <= max_cpt) & (ends > max_cpt)] = 0.0
            i += 1

        self.penalties = penalties  # Store to be able to evaluate and plot.
        return penalties

    @abc.abstractmethod
    def select_threshold(self, penalties):
        """
        threshold selection procedure.
        """

    def tune_detector(self, detector: ChangeDetector, data: pd.Series):
        penalties = self.find_penalties(detector, data.dropna())
        detector.threshold = self.select_threshold(penalties)

    # def tune_detector_collection(
    #     self, detector: DetectorCollection, data: pd.DataFrame
    # ):
    #     for name in detector.keys():
    #         self.tune_detector(detector[name], data[name])

    def tune(self, detector, data):
        if isinstance(detector, ChangeDetector):
            self.tune_detector(detector, data)
        # elif isinstance(segmentor, SegmentorCollection):
        #     self.tune_segmentor_collection(segmentor, data)
        else:
            raise ValueError(
                "tune(segmentor, data) requires segmentor to be an instance of"
                " Segmentor or SegmentorCollection."
            )

    def show(self):
        pass
