import pandas as pd
import numpy as np
from numba import njit


@njit
def window_start(window_end: int, max_window_size: int, most_recent_cpt: int = 0):
    """
    Updates the start of the window for each new observation with timestamp t.
    """
    return max(most_recent_cpt, window_end - max_window_size)


class Segmentor:
    """
    Class containing hyper-parameters for segmentation/change-point detection
    as well as generic change-point detection functionality such as
    the change-point test transformation and the change-point detection algorithm.
    """

    def __init__(
        self,
        penalty: float = None,
        min_size_window: int = 2,
        max_size_window: int = np.inf,
        max_size_history: int = np.inf,
        estimators: dict = {"mean": np.average, "sd": np.std},
    ):
        """
        Arguments:
            penalty (float): The penalty/threshold for change-point detection
                (default: 2 * log(x.size)).
            window_length (int): Maximum length in seconds of the window to
                search for change-points in.
            min_obs (int): The number of observations present in a window must
                be >= min_obs for the change-point test to be run.
            estimators (dict): Functions for estimating parameters per segment
                after change-point detection.

        Returns:
            list of changepoint timestamps (the first timestamp in each detected
            segment).
        """

        if min_size_window > max_size_window:
            raise ValueError(
                "The upper bound of the detection window must be larger than the lower"
                " bound."
            )
        if min_size_window < 2:
            raise ValueError(
                "The detection window lower bound cannot be smaller than 2."
            )

        if max_size_history < max_size_window:
            raise ValueError(
                "The maximum size of the history must be larger than the maximum size"
                " of the window."
            )

        self.penalty = penalty
        self.min_size_window = min_size_window
        self.max_size_window = max_size_window
        self.max_size_history = max_size_history
        self.estimators = estimators

        # Values since the last change-point, up to a maximum length of
        # max_history length must be kept in memory for estimating segment parameters.
        self.history = np.array([], dtype="float64")
        self.history_start = 0  # Tracks the start index of the history relative to all values used in the segmentation.
        self.timestamps = []  # Timestamp of each value in self.history.
        self.n = 0  # Total number of values input to the segmentation.
        self.window_end = min_size_window  # Current end-point of the test window.
        self.changepoints = [0]  # Changepoint indices relative to all input values.
        self.model = []  # Estimated model so far, with segments given by timestamps.

    def test(self, x: np.ndarray) -> np.ndarray:
        pass

    def tests_for(self, starts: list, ends: list, x: np.ndarray):
        """
        Outputs the optimal test statistic and maximising argument (changepoint)
        of each interval given by start[i]:(end[i]) in x.
        The purpose of this method is to facilitate computationally efficient
        penalty tuning for specific segmentors. I.e., it should be overwritten
        by a subclass.
        See UnivariateCusumSegmentor for example.
        """
        tests = []
        cpts = []
        for start, end in zip(starts, ends):
            test, cpt = self.test(x[start:end])
            tests.append(test)
            cpts.append(start + cpt)
        return tests, cpts

    def _window_start(self):
        """
        Updates the start of the window for each new observation with timestamp t.
        """
        return window_start(
            self.window_end, self.max_size_window, self.changepoints[-1]
        )

    def _history_slice(self, start, end):
        return self.history[start - self.history_start : end - self.history_start]

    def _timestamps_slice(self, start, end):
        return self.timestamps[start - self.history_start : end - self.history_start]

    def _timestamp_at(self, index):
        if index == self.n:
            return self.timestamps[-1] + 1
        return self.timestamps[index - self.history_start]

    def _append(self, values: np.ndarray, timestamps: list):
        assert values.size == len(timestamps)
        assert not pd.isnull(values).any()
        if len(self.timestamps) > 0:
            assert timestamps[0] > self.timestamps[-1]

        self.n += values.size
        self.history = np.concatenate((self.history, values))
        self.timestamps += timestamps
        return self

    def _update_changepoints(self):
        while self.window_end <= self.n:
            window_start = self._window_start()
            window_values = self._history_slice(window_start, self.window_end)
            cpt_test, cpt_in_window = self.test(window_values)
            if cpt_test > self.penalty:
                self.changepoints.append(window_start + cpt_in_window)
                self.window_end = self.changepoints[-1] + self.min_size_window
            else:
                self.window_end += 1
        return self

    def _summarise_history(self) -> list:
        history_model = []
        new_cpts = [cpt for cpt in self.changepoints if cpt > self.history_start]
        starts = [self.history_start] + new_cpts
        ends = new_cpts + [self.n]
        for start, end in zip(starts, ends):
            segment_summary = {
                "start": self._timestamp_at(start),
                "end": self._timestamp_at(end),
            }
            for name, estimator in self.estimators.items():
                segment_summary[name] = estimator(self._history_slice(start, end))
            history_model.append(segment_summary)
        return history_model

    def _update_model(self):
        history_model = self._summarise_history()
        if len(self.model) > 0:
            if self.model[-1]["start"] < self.timestamps[0]:
                history_model[0]["start"] = self.model[-1]["start"]

        self.model = self.model[:-1] + history_model
        return self

    def _shift_history(self):
        history_overshoot = self.history.size - self.max_size_history
        cpt_detected = self.changepoints[-1] > self.history_start
        if cpt_detected or (history_overshoot > 0):
            new_start = max(
                self.changepoints[-1], self.history_start + history_overshoot
            )
            self.history = self._history_slice(new_start, self.n)
            self.timestamps = self._timestamps_slice(new_start, self.n)
            self.history_start = new_start
        return self

    def update_fit_np_list(self, values: np.ndarray, timestamps: list):
        self._append(values, timestamps)
        self._update_changepoints()._update_model()._shift_history()
        return self

    def update_fit_float_int(self, value: float, timestamp: int):
        self.update_fit_np_list(np.array([value]), [timestamp])
        return self

    def update_fit_pdseries(self, series: pd.Series):
        self.update_fit_np_list(series.values, series.index.to_list())
        return self

    def update_fit(self, values, timestamps=None):
        if isinstance(values, np.ndarray) and isinstance(timestamps, list):
            self.update_fit_np_list(values, timestamps)
        elif isinstance(values, float) and isinstance(timestamps, int):
            self.update_fit_float_int(values, timestamps)
        elif isinstance(values, pd.Series):
            self.update_fit_pdseries(values)
        else:
            raise ValueError(
                "values and timestamps must be (numpy.ndarray, list), (float, int), or"
                " values must be a pd.Series with timestamps as the index."
            )
        return self

    def _format_model_output(self, model_list, times) -> pd.DataFrame:
        output_index = pd.Index(times, name="time")
        return pd.DataFrame(model_list, index=output_index).add_prefix("segment_")

    def model_at(self, times: list) -> pd.DataFrame:
        times.sort()
        output = []
        time_index = 0
        segment_index = 0

        while (time_index < len(times)) and (segment_index < len(self.model)):
            current_time = times[time_index]
            current_segment = self.model[segment_index]

            # Model evaluation is allowed before the first observed timestamp and
            # after the last observed timestamp. The first segment is extrapolated
            # into the past, while the last segment is extrapolated into the future.
            # This behaviour accomodates missing values in multivariate observations
            # when a SegmentorCollection is used.
            if current_time < self.model[0]["start"]:
                output.append(self.model[0])
                time_index += 1
            elif current_time >= self.model[-1]["end"]:
                output.append(self.model[-1])
                time_index += 1
            elif current_time < current_segment["start"]:
                time_index += 1
            elif current_time >= current_segment["end"]:
                segment_index += 1
            else:
                output.append(current_segment)
                time_index += 1

        return self._format_model_output(output, times)

    def model_now(self, parameter: str) -> float:
        return self.model[-1][parameter]

    def changepoint_timestamps(self):
        return [segment["start"] for segment in self.model]
