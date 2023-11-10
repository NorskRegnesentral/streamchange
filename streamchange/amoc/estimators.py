import abc
from numbers import Number
from typing import Tuple
import numpy as np
from numba import njit

from ..penalties import BasePenalty, ConstantPenalty, BIC


@njit
def univariate_cusum_transform(x: np.ndarray, t: np.ndarray):
    n = x.size
    sums = x.cumsum()
    return np.sqrt(n / (t * (n - t))) * (t / n * sums[-1] - sums[t - 1])


@njit
def cusum_transform(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    cusum = np.zeros((t.size, x.shape[1]))
    for j in range(x.shape[1]):
        cusum[:, j] = univariate_cusum_transform(x[:, j], t)
    return cusum


@njit
def univariate_cusum0_transform(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.cumsum(x)[t - 1] / np.sqrt(t)


@njit
def cusum0_transform(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    cusum0 = np.zeros((t.size, x.shape[1]))
    for j in range(x.shape[1]):
        cusum0[:, j] = univariate_cusum0_transform(x[:, j], t)
    return cusum0


@njit
def _optim(cpt_scores: np.ndarray, t: np.ndarray):
    argmax = cpt_scores.argmax()
    return cpt_scores[argmax], t[argmax]


@njit
def optim_univariate_cusum(x: np.ndarray, t: np.ndarray):
    cusum = univariate_cusum_transform(x, t)
    return _optim(cusum**2, t)


@njit
def optim_univariate_cusum0(x: np.ndarray, t: np.ndarray):
    cusum0 = univariate_cusum0_transform(x, t)
    return _optim(cusum0**2, t)


@njit
def optim_sum_cusum0(x: np.ndarray, t: np.ndarray):
    cusum0 = cusum0_transform(x, t)
    agg_cusum0 = (cusum0**2).sum(axis=1)
    return _optim(agg_cusum0, t)


@njit
def optim_sum_cusum(x: np.ndarray, t: np.ndarray):
    cusum = cusum_transform(x, t)
    agg_cusum = (cusum**2).sum(axis=1)
    return _optim(agg_cusum, t)


@njit
def optim_max_cusum(x: np.ndarray, t: np.ndarray):
    cusum = cusum_transform(x, t)
    cusum2 = cusum**2
    agg_cusum = np.zeros(t.size)
    for i in range(t.size):
        agg_cusum[i] = cusum2[i, :].max()
    # TODO: When njit-able: agg_cusum = np.abs(cusum).max(axis=1).
    return _optim(agg_cusum, t)


class BaseAMOCEstimator:
    _minsl_before = 1
    _minsl_after = 1

    def __init__(self, penalty: Tuple[BasePenalty, Number]):
        if isinstance(penalty, Number):
            penalty = ConstantPenalty(penalty)
        self.penalty = penalty
        self.reset()

    def reset(self):
        self._score = -np.inf
        self._changepoint = None

    @property
    def change_detected(self):
        return self._score > 0

    @property
    def score(self) -> float:
        return self._score

    @property
    def changepoint(self):
        """The most likely location of a single changepoint.

        Changepoints are consistently stored as their negative index within the
        current window. This makes it easy to extract changepoints also outside
        this class, where the relevant temporal frame of reference is.
        """
        return self._changepoint

    @abc.abstractmethod
    def _fit(
        self,
        x: np.ndarray,
        candidate_cpts: np.ndarray = None,
    ) -> Tuple[float, int]:
        """Subclass-specific method for detecting a single changepoint"""

    def fit(
        self, x: np.ndarray, candidate_cpts: np.ndarray = None
    ) -> "BaseAMOCEstimator":
        """Detect whether there is at least one changepoint in a data vector.

        Should set the self._change_detected and self._changepoint variables.

        Parameters
        ----------
        x :
            Input values.

        candidate_cpts :
            Sorted, 1-dimensional numpy.ndarray of candidate change-points within x.

        Returns
        -------
        self

        """
        self.reset()
        if candidate_cpts is None:
            n = x.shape[0]
            min_candidate = self._minsl_after
            max_candidate = n - self._minsl_before + 1
            candidate_cpts = np.arange(min_candidate, max_candidate)
        if candidate_cpts.size > 0:
            # To not clutter WindowSegmentor, it is convenient to allow size = 0
            # input of candidate_cpts.
            self._score, self._changepoint = self._fit(x, candidate_cpts)
        return self


class SeparableAMOCEstimator(BaseAMOCEstimator):
    def __init__(self, penalty: Tuple[ConstantPenalty, Number] = BIC()):
        super().__init__(penalty)

    def reset(self):
        super().reset()
        self._raw_score = 0.0

    @property
    def raw_score(self) -> float:
        return self._raw_score

    @staticmethod
    @abc.abstractmethod
    def _changepoint_optimiser(x, candidate_cpts):
        pass

    def _fit(self, x, candidate_cpts):
        self._raw_score, cpt = self._changepoint_optimiser(x, candidate_cpts)
        return self.raw_score - self.penalty(), cpt


class CUSUM(SeparableAMOCEstimator):
    @staticmethod
    def _changepoint_optimiser(x, candidate_cpts):
        return optim_univariate_cusum(x, candidate_cpts)


class CUSUM0(SeparableAMOCEstimator):
    _minsl_before = 0

    @staticmethod
    def _changepoint_optimiser(x, candidate_cpts):
        return optim_univariate_cusum0(x, candidate_cpts)


class SumCUSUM0(SeparableAMOCEstimator):
    _minsl_before = 0

    @staticmethod
    def _changepoint_optimiser(x, candidate_cpts):
        return optim_sum_cusum0(x, candidate_cpts)


class SumCUSUM(SeparableAMOCEstimator):
    @staticmethod
    def _changepoint_optimiser(x, candidate_cpts):
        return optim_sum_cusum(x, candidate_cpts)


class MaxCUSUM(SeparableAMOCEstimator):
    @staticmethod
    def _changepoint_optimiser(x, candidate_cpts):
        return optim_max_cusum(x, candidate_cpts)
