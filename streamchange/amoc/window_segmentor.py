import numpy as np
import pandas as pd
from river.stream import iter_pandas

from ..base import ChangeDetector, NumpyDeque
from ..utils import geomspace_int
from .estimators import BaseAMOCEstimator

# TODO: Add fit and predict like in Capa.


class WindowSegmentor(ChangeDetector):
    """
    Class for testing-based changepoint detection.

    Parameters
    ----------
    estimator
        The single changepoint test.
    min_window
        The minimum length of the window to test for changepoints in.
    max_window
        The maximum length of the window to test for changepoints in.
    minsl
        The minimum segment length.
    candidate_type
        The type of candidate changepoints set. Must be either "linear" or "geom".
    candidate_step
        The step size of the candidate changepoint set. If candidate_type=="geom",
        the step size is the factor to multiply the previous changepoint with to
        generate the set, and it must therefore be > 1.
    with_jumpback
        Upon detection of a changepoint, whether to jump back to a minimum
        length window starting right after the changepoints.
    """

    def __init__(
        self,
        estimator: BaseAMOCEstimator,
        min_window: int = 2,
        max_window: int = np.inf,
        minsl: int = 1,
        candidate_type: str = "linear",
        candidate_step: float = 1,
        with_jumpback: bool = True,
    ):
        self.estimator = estimator
        self.min_window, self.max_window, self.minsl = self._validate_window(
            min_window, max_window, minsl
        )
        self.candidate_type = candidate_type
        self.candidate_step = candidate_step
        self.with_jumpback = with_jumpback
        self.candidate_cpts = self._make_candidate_cpts()
        self.window = NumpyDeque(max_window)
        self.reset()

    def _reset_changepoints(self):
        super().reset()

    def reset(self) -> "WindowSegmentor":
        self._reset_changepoints()
        self.estimator.reset()
        self.window.reset()
        return self

    def _validate_window(self, min_window, max_window, minsl):
        if min_window < 2:
            raise ValueError("min_window cannot be smaller than 2.")
        if min_window > max_window:
            raise ValueError("min_window cannot be greater than max_window.")
        if minsl < max(self.estimator._minsl_before, self.estimator._minsl_after):
            message = "minsl cannot be smaller than the strictest minsl restriction in the AMOC estimator."
            raise ValueError(message)
        if minsl > max_window:
            raise ValueError("minsl cannot be greater than max_window.")

        return min_window, max_window, minsl

    def _make_candidate_cpts(self):
        # Candidate changepoints only run till n-1 to avoid the same changepoint
        # being tested twice, which could result in an infinite loop in .update()
        min_cpt = self.minsl if self.estimator._minsl_after > 0 else 0
        max_cpt = (
            self.max_window - self.minsl
            if self.estimator._minsl_before > 0
            else self.max_window
        )
        if max_cpt < min_cpt:
            message = "minsl cannot be greater than max_window/2 when the AMOC estimator requires estimation both before and after a candidate changepoint."
            raise ValueError(message)

        if self.candidate_type == "linear":
            candidate_cpts = np.arange(min_cpt, max_cpt + 1, self.candidate_step)
        elif self.candidate_type == "geom":
            if self.candidate_step <= 1.0:
                message = "When candidate_type='geom', candidate_step must be > 1."
                raise ValueError(message)
            candidate_cpts = geomspace_int(min_cpt, max_cpt + 1, self.candidate_step)
        else:
            raise ValueError("candidate_type must be either 'linear' or 'geom'.")

        return candidate_cpts

    def update(self, x):
        if self.change_detected:
            self.window.pop(len(self.window) - self.changepoints[-1])
        self.window.appendleft(x)
        self._reset_changepoints()

        start = len(self.window)
        end = min(0, start - self.min_window)
        while end >= 0:
            window_values = self.window.values[end:start]
            n = window_values.shape[0]
            candidate_cpts = self.candidate_cpts[self.candidate_cpts < n]
            self.estimator.fit(window_values, candidate_cpts)
            if self.estimator.change_detected:
                cpt = self.estimator.changepoint
                self._changepoints.append(cpt)
                if self.with_jumpback:
                    start = cpt
                    end = start - self.min_window + 1
            end -= 1
        return self

    def fit(self, x: pd.DataFrame = None) -> "WindowSegmentor":
        return self

    def predict(self, x: pd.DataFrame) -> list:
        self.reset()
        x = x.dropna()
        times = x.index
        x = x.reset_index(drop=True)
        cpts = []
        for t, x_t in x.to_dict(orient="index").items():
            self.update(x_t)
            if self.change_detected:
                cpts += [t - cpt for cpt in self.changepoints]
        return times[cpts].tolist()

    def fit_predict(self, x: pd.DataFrame) -> list:
        return self.fit(x).predict(x)
