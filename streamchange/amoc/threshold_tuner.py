import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go

from .window_segmentor import WindowSegmentor


@njit
def generate_intervals(
    data_size: int,
    min_window: int,
    max_window: int,
    prob: float = 1.0,
):
    # TODO: Account for minsl in interval generation.
    starts = []
    ends = []
    for end in range(min_window, data_size):
        min_start = max(0, end - max_window)
        max_start = end - min_window + 1
        for start in range(min_start, max_start):
            if np.random.uniform(0.0, 1.0) <= prob:
                starts.append(start)
                ends.append(end)
    return np.array(starts), np.array(ends)


def base_selector(alpha=0.0):
    def selector(thresholds):
        return max((1 + alpha) * thresholds[-1], 1e-8)

    return selector


# TODO: Update ThresholdTuner with new cpt definition and new WindowSegmentor


class ThresholdTuner:
    """
    Class for tuning WindowSegmentor detectors that use amoc tests with a single threshold.
    """

    def __init__(
        self,
        max_cpts: int = 1000,
        prob: float = 1.0,
        max_window_only=True,
        selector=base_selector(),
    ):
        self.max_cpts = max_cpts
        self.prob = prob
        self.max_window_only = max_window_only
        self.selector = selector

    def _detect_in(self, starts: list, ends: list):
        """
        Outputs the optimal test statistic and changepoint of each interval given
        by start[i]:(end[i]) in x.
        """
        scores = []
        cpts = []
        for start, end in zip(starts, ends):
            self.detector.estimator.fit(self.x[start:end])
            scores.append(self.detector.estimator.score)
            cpts.append(end + self.detector.estimator.changepoint)
        return np.array(scores), np.array(cpts)

    def _find_thresholds(self) -> np.ndarray:
        max_window = self.detector.max_window
        min_window = max_window if self.max_window_only else self.detector.min_window
        n = self.x.shape[0]
        starts, ends = generate_intervals(n, min_window, max_window, self.prob)
        scores, cpts = self._detect_in(starts, ends)

        self.thresholds = np.zeros(self.max_cpts)
        i = 0
        while (i < self.max_cpts) & np.any(scores > 0.0):
            argmax = scores.argmax()
            self.thresholds[i] = scores[argmax]
            max_cpt = cpts[argmax]
            cpt_in_interval = (max_cpt >= starts) & (max_cpt < ends)
            scores[cpt_in_interval] = 0.0
            i += 1

    def tune(self, detector: WindowSegmentor, x: pd.DataFrame):
        assert x.shape[0] >= self.max_cpts

        self.detector = detector
        self.x = x.to_numpy()
        self._find_thresholds()
        detector.estimator.penalty = self.selector(self.thresholds)

    def __call__(self, detector: WindowSegmentor, data: pd.DataFrame):
        self.tune(detector, data)

    def show(self, title="") -> go:
        fig = go.Figure(
            layout=go.Layout(
                title=title,
                xaxis_title="Number of changepoints",
                yaxis_title="Threshold",
            )
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.thresholds,
                    mode="markers",
                    name="Threshold series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=np.repeat(self.detector.estimator.penalty, self.max_cpts),
                    mode="lines",
                    line_dash="dot",
                    name="Tuned threshold",
                ),
            ]
        )
        return fig
