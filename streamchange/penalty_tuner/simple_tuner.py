from . import PenaltyTuner

import numpy as np
import plotly.graph_objects as go


class SimpleTuner(PenaltyTuner):
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
