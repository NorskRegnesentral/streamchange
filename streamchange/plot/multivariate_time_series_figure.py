import numpy as np
import pandas as pd

from . import TimeSeriesFigure
from plotly.subplots import make_subplots


class MultivariateTimeSeriesFigure(TimeSeriesFigure):
    def __init__(self, row_names, title="", xaxis="Timestamp", yaxis="Value"):
        self._rows = {name: row + 1 for row, name in enumerate(row_names)}
        subplots = make_subplots(
            rows=len(self._rows),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            x_title=xaxis,
        )
        self.__dict__.update(subplots.__dict__)
        self.update_layout(
            template="simple_white",
            title_text=title,
        )

        fig_boundaries = np.linspace(0, 1, len(self._rows) + 1)[::-1]
        for name, row in self._rows.items():
            y = (
                fig_boundaries[row - 1]
                - (fig_boundaries[row - 1] - fig_boundaries[row]) / 2
            )
            shown_name = str(name)
            if len(shown_name) > 10:
                shown_name = f"{name[:10]}..."
            self.add_annotation(
                showarrow=False,
                text=shown_name,
                textangle=90,
                xanchor="left",
                xref="paper",
                x=-0.085,
                y=y,
                yanchor="middle",
                yref="paper",
            )

        self._raw_data_legend_added = False
        self._mean_legend_added = False
        self._confidence_interval_legend_added = False
        self._highlight_added = False

    def add_raw_data_at(self, raw_values: pd.Series, name: str):
        row = self._rows[name]
        trace = self.make_raw_data_trace(raw_values)
        if self._raw_data_legend_added:
            trace.showlegend = False
        else:
            self._raw_data_legend_added = True
        self.add_trace(trace, row=row, col=1)

    def add_mean_at(self, mean_values: pd.Series, name: str):
        row = self._rows[name]
        trace = self.make_mean_trace(mean_values)
        if self._mean_legend_added:
            trace.showlegend = False
        else:
            self._mean_legend_added = True
        self.add_trace(trace, row=row, col=1)

    def add_highlighted_values_at(
        self, values: pd.Series, name: str, trace_name="Highlighted"
    ):
        row = self._rows[name]
        trace = self.make_highlight_trace(values, trace_name)
        if self._highlight_added:
            trace.showlegend = False
        else:
            self._highlight_added = True
        self.add_trace(trace, row=row, col=1)

    def add_confidence_band_at(
        self,
        ci_lower: pd.Series,
        ci_upper: pd.Series,
        name: str,
        confidence_level: float = None,
    ):
        row = self._rows[name]
        traces = self.make_confidence_band_traces(ci_lower, ci_upper, confidence_level)
        if self._confidence_interval_legend_added:
            for trace in traces:
                trace.showlegend = False
        else:
            self._confidence_interval_legend_added = True
        for trace in traces:
            self.add_trace(trace, row=row, col=1)

    def add_raw_data(self, raw_values: pd.DataFrame):
        for name, series in raw_values.items():
            self.add_raw_data_at(series, name)

    def add_mean(self, mean_values: pd.DataFrame):
        for name, series in mean_values.items():
            self.add_mean_at(series, name)

    def add_highlighted_values(self, values: pd.DataFrame, trace_name="Highlighted"):
        for name, series in values.items():
            self.add_highlighted_values_at(series, name, trace_name)

    def add_confidence_band(
        self,
        ci_lower: pd.DataFrame,
        ci_upper: pd.DataFrame,
        confidence_level: float = None,
    ):
        for col in ci_lower:
            self.add_confidence_band_at(
                ci_lower[col], ci_upper[col], col, confidence_level
            )
