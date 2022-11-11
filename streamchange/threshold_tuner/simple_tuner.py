from . import ThresholdTuner

import numpy as np
import plotly.graph_objects as go


class SimpleTuner(ThresholdTuner):
    def __init__(self, alpha: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def select_threshold(self, penalties):
        # Store for use in show/plotting method.
        self.threshold = max((1 + self.alpha) * penalties[-1], 1e-8)
        return self.threshold

    def show(self) -> go:
        fig = go.Figure(
            layout=go.Layout(
                title=f"Threshold = (1+{self.alpha}) * last threshold in series",
                xaxis_title="Number of changepoints",
                yaxis_title="Threshold",
            )
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.penalties,
                    mode="markers",
                    name="Threshold series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=np.repeat(self.threshold, self.max_cpts),
                    mode="lines",
                    line_dash="dot",
                    name="Tuned threshold",
                ),
            ]
        )
        return fig
