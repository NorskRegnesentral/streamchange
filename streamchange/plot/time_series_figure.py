import pandas as pd
import plotly.graph_objects as go


class TimeSeriesFigure(go.Figure):
    def __init__(self, title="", xaxis="Timestamp", yaxis="Value"):
        super().__init__(
            layout=go.Layout(
                template="simple_white",
                title=title,
                xaxis_title=xaxis,
                yaxis_title=yaxis,
            )
        )

    @staticmethod
    def make_raw_data_trace(raw_values: pd.Series):
        return go.Scatter(
            x=raw_values.index,
            y=raw_values,
            mode="markers",
            marker_size=1.5,
            marker_color="black",
            name="Raw data",
        )

    @staticmethod
    def make_mean_trace(mean_values: pd.Series):
        return go.Scatter(
            x=mean_values.index,
            y=mean_values,
            mode="markers",
            marker_size=2,
            marker_color="steelblue",
            name="Mean",
        )

    @staticmethod
    def make_highlight_trace(values: pd.Series, name="Highlighted"):
        return go.Scatter(
            x=values.index,
            y=values,
            mode="markers",
            marker_size=3,
            marker_color="red",
            name=name,
        )

    @staticmethod
    def make_confidence_band_traces(
        ci_lower: pd.Series, ci_upper: pd.Series, confidence_level: float = None
    ):
        if confidence_level is None:
            legend_text = f"Confidence interval"
        else:
            legend_text = f"{100*confidence_level}% confidence interval"

        return [
            go.Scatter(
                x=ci_lower.index,
                y=ci_lower,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
                name="Upper confidence bound",
            ),
            go.Scatter(
                x=ci_upper.index,
                y=ci_upper,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name=legend_text,
                fill="tonexty",
                fillcolor="rgba(70, 130, 180, 0.4)",  # Steel blue wih 0.4 opacity.
            ),
        ]

    def add_raw_data(self, raw_values: pd.Series):
        self.add_trace(self.make_raw_data_trace(raw_values))

    def add_mean(self, mean_values: pd.Series):
        self.add_trace(self.make_mean_trace(mean_values))

    def add_highlighted_values(self, values: pd.Series, trace_name="Highlighted"):
        self.add_trace(self.make_highlight_trace(values, trace_name))

    def add_confidence_band(
        self, ci_lower: pd.Series, ci_upper: pd.Series, confidence_level: float = None
    ):
        self.add_traces(
            self.make_confidence_band_traces(ci_lower, ci_upper, confidence_level)
        )
