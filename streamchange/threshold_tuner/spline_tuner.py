from . import ThresholdTuner
from .utils import signed_curvature

import numpy as np
from scipy.interpolate import UnivariateSpline
import plotly.graph_objects as go


class SplineTuner(ThresholdTuner):
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

    def select_threshold(self, penalties):
        self.normalised_penalties = self.max_cpts * self._normalise(penalties)
        self.smooth_penalties = self._spline_smooth(
            self.normalised_penalties, self.max_cpts * self.smoothing_factor
        )
        argmax_curvature = signed_curvature(self.smooth_penalties).argmax()
        # The threshold can be a bit unstable if the data is completely constant.
        # I.e., it might get values on the order of 1e-15 due to rounding errors.
        # Capping downward at 1e-8 is a reasonable choice.
        self.threshold = max(  # Store for use in show/plotting method.
            penalties[argmax_curvature], (1 + 0.01) * penalties[-1], 1e-8
        )
        return self.threshold

    def show(self) -> go:
        min_pen = self.penalties.min()
        max_pen = self.penalties.max()
        normalised_threshold = (
            self.max_cpts * (self.threshold - min_pen) / (max_pen - min_pen)
        )

        fig = go.Figure(
            layout=go.Layout(
                title=(
                    "Threshold = point of maximum curvature of smoothed normalised"
                    " threshold series"
                ),
                xaxis_title="Number of changepoints",
                yaxis_title="Normalised threshold",
            )
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.normalised_penalties,
                    mode="markers",
                    name="Normalised threshold series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.smooth_penalties,
                    mode="lines",
                    name="Spline-smoothed threshold series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=np.repeat(normalised_threshold, self.max_cpts),
                    mode="lines",
                    line_dash="dot",
                    name="Tuned threshold",
                ),
            ]
        )
        return fig
