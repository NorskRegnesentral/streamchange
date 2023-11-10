import abc
import copy
from numbers import Number
import numpy as np
import pandas as pd
from typing import Tuple, Union, Callable

from ..penalties import BasePenalty, ConstantPenalty
from ..segment_stats import MovingSum


class BaseScore:
    @abc.abstractmethod
    def reset(self) -> "BaseScore":
        return self

    @property
    @abc.abstractmethod
    def value(self) -> float:
        pass

    @abc.abstractmethod
    def update(self, x: Union[Number, np.ndarray]) -> "BaseScore":
        """Update the score with a single data point.

        Parameters
        ----------
        x
            One observation row-vector.

        Returns
        -------
        self
        """
        return self

    def fit(self, x: pd.DataFrame) -> "BaseScore":
        self.reset()
        x = x.dropna()
        self.values_ = []
        for x_t in x.values:
            self.update(x_t)
            self.values_.append(self.value)
        self.values_ = pd.Series(self.values_, index=x.index)
        return self

    @property
    def changepoint(self):
        """Optional property for those scores that also give esimates of the most recent changepoint location per iteration."""
        return None


class BaseRawScore(BaseScore):
    """A raw changepoint score. It is always >= 0"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._score = 0.0
        return self

    @property
    def value(self):
        return float(self._score)

    def penalise(self, penalty: BasePenalty):
        return PenalisedScore(self, penalty)


class BasePenalisedScore(BaseScore):
    """A penalised changepoint score where self.value > 0 means a change is detected."""

    def __init__(self, penalty: Tuple[BasePenalty, Number]):
        if isinstance(penalty, Number):
            penalty = ConstantPenalty(penalty)
        self.penalty = penalty
        self.reset()

    def reset(self):
        self._penalised_score = -self.penalty()
        return self

    @property
    def value(self):
        """Get the current value of the penalised score."""
        return float(self._penalised_score)


class PenalisedScore(BasePenalisedScore):
    def __init__(self, score: BaseRawScore, penalty: BasePenalty):
        self.score = score
        super().__init__(penalty)
        self.reset()

    def reset(self):
        self.score.reset()
        super().reset()
        return self

    def update(self, x):
        self.score.update(x)
        self._penalised_score = self.score.value - self.penalty()
        return self

    @property
    def changepoint(self):
        return self.score.changepoint


class AggregatedScore(BaseRawScore):
    def __init__(self, base_score: BaseRawScore, aggregator: Callable = sum):
        self.base_score = base_score
        self.aggregator = aggregator
        self.reset()

    def reset(self):
        self.scores = None
        super().reset()
        return self

    def _init_scores(self, x):
        self.scores = [copy.deepcopy(self.base_score) for _ in x]

    def update(self, x: list):
        if self.scores is None:
            self._init_scores(x)

        for i, x_i in enumerate(x):
            self.scores[i].update(x_i)

        self._score = self.aggregator([score.value for score in self.scores])
        return self

    # TODO: Dict implementation????
    # def _init_scores(self, x: dict):
    #     self.scores = {key: copy.deepcopy(self.base_score) for key, _ in x.items()}

    # def update(self, x: dict):
    #     if self.scores is None:
    #         self._init_scores(x)

    #     for key, value in x.items():
    #         self.scores[key].update(value)

    #     self._score = self.aggregator([score.value for score in self.scores])
    #     return self


class LordenPollakScore(BaseRawScore):
    def __init__(self, rho: Number = 1.0):
        self.rho = rho
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        super().reset()
        return self

    def update(self, x: Number):
        mean = self.sum / self.n if self.n > 0 else 0
        mu = max(mean, self.rho)
        self._score = max(0, self._score + mu * x - mu**2 / 2)
        if self._score < 1e-8:
            self.reset()
        else:
            self.n += 1
            self.sum += x
        return self

    @property
    def changepoint(self):
        return self.n + 1


class CUSUM0Score(BaseRawScore):
    def __init__(self, window_sizes: list = [2, 5, 10, 50, 100]):
        self.window_sizes = window_sizes
        self.weights = [1 / window_size for window_size in self.window_sizes]
        self.reset()

    def reset(self):
        self.sums = [
            MovingSum(window_size=window_size) for window_size in self.window_sizes
        ]
        super().reset()
        return self

    def update(self, x: Number):
        for s in self.sums:
            s.update(x)

        self.cusum = [w * s.value**2 for s, w in zip(self.sums, self.weights)]
        self._score = max(self.cusum)

    def changepoint(self):
        return self.window_sizes[np.argmax(self.cusum)]


# from ..base import NumpyDeque
# from ..utils import geomspace_int
# from ..amoc import BaseAMOCEstimator


# class AMOCScore(BasePenalisedScore):
#     """
#     Class for testing-based changepoint detection.

#     Parameters
#     ----------
#     estimator
#         The single changepoint test.
#     min_window
#         The minimum length of the window to test for changepoints in.
#     max_window
#         The maximum length of the window to test for changepoints in.
#     minsl
#         The minimum segment length.
#     candidate_type
#         The type of candidate changepoints set. Must be either "linear" or "geom".
#     candidate_step
#         The step size of the candidate changepoint set. If candidate_type=="geom",
#         the step size is the factor to multiply the previous changepoint with to
#         generate the set, and it must therefore be > 1.
#     """

#     def __init__(
#         self,
#         estimator: BaseAMOCEstimator,
#         min_window: int = 2,
#         max_window: int = int(1e5),
#         minsl: int = 1,
#         candidate_type: str = "linear",
#         candidate_step: float = 1,
#     ):
#         self.estimator = estimator
#         self._validate_window(min_window, max_window, minsl)
#         self.min_window = min_window
#         self.max_window = max_window
#         self.minsl = minsl
#         self.candidate_type = candidate_type
#         self.candidate_step = candidate_step
#         self.candidate_cpts = self._make_candidate_cpts()
#         self.window = NumpyDeque(max_window)
#         self.reset()

#     def reset(self):
#         super().reset()
#         self.last_changepoint = 0
#         self.estimator.reset()
#         self.window.reset()
#         return self

#     def get_penalty(self):
#         return self.estimator.penalty

#     def _validate_window(self, min_window, max_window, minsl):
#         if min_window < 2:
#             raise ValueError("min_window cannot be smaller than 2.")
#         if min_window > max_window:
#             raise ValueError("min_window cannot be greater than max_window.")
#         if minsl < max(self.estimator._minsl_before, self.estimator._minsl_after):
#             msg = "minsl cannot be smaller than the strictest minsl restriction in the AMOC estimator."
#             raise ValueError(msg)

#         is_onesided_estimator = (
#             self.estimator._minsl_after == 0 or self.estimator._minsl_before == 0
#         )
#         if is_onesided_estimator and minsl > max_window:
#             msg = "minsl cannot be greater than max_window for one-sided AMOC estimators. "
#             raise ValueError(msg)
#         elif not is_onesided_estimator and minsl > max_window / 2:
#             msg = "minsl cannot be greater than max_window/2 for two-sided AMOC estimators."
#             raise ValueError(msg)

#     def _make_candidate_cpts(self):
#         # Candidate changepoints only run till n-1 to avoid the same changepoint
#         # being tested twice, which could result in an infinite loop in .update()
#         min_cpt = self.minsl if self.estimator._minsl_after > 0 else 0
#         max_cpt = (
#             self.max_window - self.minsl
#             if self.estimator._minsl_before > 0
#             else self.max_window
#         )
#         if max_cpt < min_cpt:
#             message = "minsl cannot be greater than max_window/2 when the AMOC estimator requires estimation both before and after a candidate changepoint."
#             raise ValueError(message)

#         if self.candidate_type == "linear":
#             candidate_cpts = np.arange(min_cpt, max_cpt + 1, self.candidate_step)
#         elif self.candidate_type == "geom":
#             if self.candidate_step <= 1.0:
#                 message = "When candidate_type='geom', candidate_step must be > 1."
#                 raise ValueError(message)
#             candidate_cpts = geomspace_int(min_cpt, max_cpt + 1, self.candidate_step)
#         else:
#             raise ValueError("candidate_type must be either 'linear' or 'geom'.")

#         return candidate_cpts

#     def _get_valid_candidate_cpts(self):
#         minsl_boundary = self.last_changepoint - self.minsl + 1
#         valid_candidates = self.candidate_cpts < min(minsl_boundary, len(self.window))
#         return self.candidate_cpts[valid_candidates]

#     @property
#     def changepoint(self):
#         return self.estimator.changepoint

#     def update(self, x):
#         self.estimator.reset()
#         self.window.appendleft(x)
#         # Need some upper limit on the last changepoint index.
#         self.last_changepoint = min(self.last_changepoint + 1, int(1e8))

#         if len(self.window) >= self.min_window:
#             candidate_cpts = self._get_valid_candidate_cpts()
#             self.estimator.fit(self.window.values, candidate_cpts)
#             self._penalised_score = self.estimator.score
#             if self.estimator.change_detected:
#                 self.last_changepoint = self.changepoint
#                 self.window.pop(len(self.window) - self.changepoint)

#         return self
