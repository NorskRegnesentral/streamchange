from .segmentor import Segmentor, window_start
from .segmentor_collection import SegmentorCollection
from .utils import signed_curvature

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go
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


class BasicPenaltyTuner(PenaltyTuner):
    def __init__(self, alpha: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def select_penalty(self, penalties):
        # Store for use in show/plotting method.
        self.penalty = max((1 + self.alpha) * penalties[-1], 1e-8)
        return self.penalty

    def show(self) -> go:
        fig = go.Figure(
            layout=go.Layout(
                title=f"Penalty = (1+{self.alpha}) * last penalty in series",
                xaxis_title="Number of changepoints",
                yaxis_title="Penalty",
            )
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.penalties,
                    mode="markers",
                    name="Penalty series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=np.repeat(self.penalty, self.max_cpts),
                    mode="lines",
                    line_dash="dot",
                    name="Tuned penalty",
                ),
            ]
        )
        return fig


class SplinePenaltyTuner(PenaltyTuner):
    def __init__(self, smoothing_factor: int = 50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing_factor = smoothing_factor

    @staticmethod
    def _spline_smooth(x: np.ndarray, smoothing_factor: float) -> np.ndarray:
        k = np.arange(x.size)
        spline = UnivariateSpline(k, x)
        spline.set_smoothing_factor(smoothing_factor)
        return spline(k)

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        return (x - x.min()) / (x.max() - x.min())

    def select_penalty(self, penalties):
        self.normalised_penalties = self.max_cpts * self._normalise(penalties)
        self.smooth_penalties = self._spline_smooth(
            self.normalised_penalties, self.max_cpts * self.smoothing_factor
        )
        argmax_curvature = signed_curvature(self.smooth_penalties).argmax()
        # The penalty can be a bit unstable if the data is completely constant.
        # I.e., it might get values on the order of 1e-15 due to rounding errors.
        # Capping downward at 1e-8 is a reasonable choice.
        self.penalty = max(  # Store for use in show/plotting method.
            penalties[argmax_curvature], (1 + 0.01) * penalties[-1], 1e-8
        )
        return self.penalty

    def show(self) -> go:
        min_pen = self.penalties.min()
        max_pen = self.penalties.max()
        normalised_penalty = (
            self.max_cpts * (self.penalty - min_pen) / (max_pen - min_pen)
        )

        fig = go.Figure(
            layout=go.Layout(
                title=(
                    "Penalty = point of maximum curvature of smoothed normalised"
                    " penalty series"
                ),
                xaxis_title="Number of changepoints",
                yaxis_title="Normalised penalty",
            )
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.normalised_penalties,
                    mode="markers",
                    name="Normalised penalty series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.smooth_penalties,
                    mode="lines",
                    name="Spline-smoothed penalty series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=np.repeat(normalised_penalty, self.max_cpts),
                    mode="lines",
                    line_dash="dot",
                    name="Tuned penalty",
                ),
            ]
        )
        return fig
