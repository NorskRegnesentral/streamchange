import abc
from typing import Tuple, Callable
import numbers
import numpy as np
from numba import njit


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
def _optim(cusum: np.ndarray, t: np.ndarray):
    argmax = cusum.argmax()
    return cusum[argmax], t[argmax]


@njit
def optim_univariate_cusum(x: np.ndarray, t: np.ndarray):
    cusum = univariate_cusum_transform(x, t)
    abs_cusum = np.abs(cusum)
    return _optim(abs_cusum, t)


@njit
def optim_univariate_cusum0(x: np.ndarray, t: np.ndarray):
    sums = np.cumsum(x)
    abs_cusum = np.abs(sums[t - 1] / np.sqrt(t))
    return _optim(abs_cusum, t)


@njit
def optim_sum_cusum(x: np.ndarray, t: np.ndarray):
    cusum = cusum_transform(x, t)
    agg_cusum = np.abs(cusum).sum(axis=1)
    return _optim(agg_cusum, t)


@njit
def optim_max_cusum(x: np.ndarray, t: np.ndarray):
    abs_cusum = np.abs(cusum_transform(x, t))
    agg_cusum = np.zeros(t.size)
    for i in range(t.size):
        agg_cusum[i] = abs_cusum[i, :].max()
    # TODO: When njit-able: agg_cusum = np.abs(cusum).max(axis=1).
    return _optim(agg_cusum, t)


class AMOCEstimator:
    _minsl_before = 1
    _minsl_after = 1

    def __init__(self):
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

    def fit(self, x: np.ndarray, candidate_cpts: np.ndarray = None) -> "AMOCEstimator":
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

    @abc.abstractmethod
    def _fit(
        self,
        x: np.ndarray,
        candidate_cpts: np.ndarray = None,
    ) -> Tuple[float, int]:
        """Subclass-specific method for detecting a single changepoint"""


class CUSUM(AMOCEstimator):
    def __init__(
        self,
        penalty: numbers.Number = None,
        arl: int = 10000,
        p=1,
    ):
        super().__init__()
        self.arl = arl
        self.p = p
        self.penalty = self.default_penalty(arl, p) if penalty is None else penalty

    @staticmethod
    def default_penalty(n: int, p: int = 1) -> float:
        """Default penalty as function of n and p"""
        return np.sqrt(2.0 * p * np.log(n))

    def _fit_cusum(self, x, candidate_cpts, optimiser: Callable):
        score, cpt = optimiser(x, candidate_cpts)
        score = score - self.penalty
        return score, cpt

    def _fit(self, x, candidate_cpts):
        optimiser = optim_univariate_cusum
        return self._fit_cusum(x, candidate_cpts, optimiser)


class CUSUM0(CUSUM):
    _minsl_before = 0

    def _fit(self, x, candidate_cpts):
        optimiser = optim_univariate_cusum0
        return self._fit_cusum(x, candidate_cpts, optimiser)


class SumCUSUM(CUSUM):
    def _fit(self, x, candidate_cpts):
        optimiser = optim_sum_cusum
        return self._fit_cusum(x, candidate_cpts, optimiser)


class MaxCUSUM(CUSUM):
    def _fit(self, x, candidate_cpts):
        optimiser = optim_max_cusum
        return self._fit_cusum(x, candidate_cpts, optimiser)
