import abc
from typing import Tuple
import numbers
import numpy as np
from numba import njit


class AMOCTest:
    def __init__(self):
        self.reset()

    def reset(self):
        self._change_detected = False
        self._changepoint = None

    @property
    def change_detected(self):
        return self._change_detected

    @property
    def changepoint(self):
        """The most likely location of a single changepoint.

        Changepoints are consistently stored as their negative index within the
        current window. This makes it easy to extract changepoints also outside
        this class, where the relevant temporal frame of reference is.
        """
        return self._changepoint

    @abc.abstractmethod
    def detect(self, x: np.ndarray) -> "AMOCTest":
        """Detect whether there is at least one changepoint in a data vector.

        Should set the self._change_detected and self._changepoint variables.

        Parameters
        ----------
        x
            Input values.

        Returns
        -------
        self

        """

        return self


# CUSUM
# ------------------------------------------------------------------------------
@njit
def univariate_cusum_transform(x: np.ndarray, t: np.ndarray):
    n = x.size
    sums = x.cumsum()
    return np.sqrt(n / (t * (n - t))) * (t / n * sums[-1] - sums[t - 1])


@njit
def cusum_transform(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    cusum = np.zeros((x.shape[0] - 1, x.shape[1]))
    for j in range(x.shape[1]):
        cusum[:, j] = univariate_cusum_transform(x[:, j], t)
    return cusum


@njit
def _optim(agg_cusum: np.ndarray, t: np.ndarray, n: int):
    argmax = agg_cusum.argmax()
    return agg_cusum[argmax], t[argmax] - n - 1  # Negative changepoint index.


class CUSUM(AMOCTest):
    def __init__(self, threshold: numbers.Number = 0.0, minsl: int = 1):
        super().__init__()
        assert threshold >= 0.0
        self.threshold = threshold
        self.minsl = minsl
        self.reset()

    def reset(self):
        super().reset()
        self._score = -np.inf

    @property
    def score(self) -> float:
        return self._score

    @abc.abstractmethod
    def set_default_threshold(self, n: float, p: float):
        # TODO: Handle this is in capa.
        return self

    @abc.abstractmethod
    def _detect(self, x, t) -> Tuple[float, int]:
        pass

    def detect(self, x):
        self.reset()
        if x.shape[0] >= 2 * self.minsl:
            t = np.arange(self.minsl, x.shape[0] - self.minsl + 1)
            self._score, self._changepoint = self._detect(x, t)
            self._change_detected = self._score > self.threshold
        return self


# Univariate CUSUM
# ------------------------------------------------------------------------------
@njit
def optim_univariate_cusum(x: np.ndarray, t: np.ndarray):
    cusum = univariate_cusum_transform(x, t)
    abs_cusum = np.abs(cusum)
    return _optim(abs_cusum, t, x.shape[0])


class UnivariateCUSUM(CUSUM):
    def set_default_threshold(self, n, p=1):
        self.threshold = np.sqrt(2.0 * np.log(n))
        return self

    def _detect(self, x, t):
        return optim_univariate_cusum(x, t)


# Multivariate CUSUMs
# ------------------------------------------------------------------------------
@njit
def optim_sum_cusum(x: np.ndarray, t: np.ndarray):
    cusum = cusum_transform(x, t)
    agg_cusum = np.abs(cusum).sum(axis=1)
    return _optim(agg_cusum, t, x.shape[0])


class SumCUSUM(CUSUM):
    def set_default_threshold(self, n: float, p: float):
        self.threshold = np.sqrt(2.0 * p * np.log(n))
        return self

    def _detect(self, x, t):
        return optim_sum_cusum(x, t)


@njit
def optim_max_cusum(x: np.ndarray, t: np.ndarray):
    abs_cusum = np.abs(cusum_transform(x, t))
    agg_cusum = np.zeros(abs_cusum.shape[0])
    for i in range(x.shape[0]):
        agg_cusum[i] = abs_cusum[i, :].max()
    # agg_cusum = np.abs(cusum).max(axis=1)
    return _optim(agg_cusum, t, x.shape[0])


class MaxCUSUM(CUSUM):
    def set_default_threshold(self, n: float, p: float):
        self.threshold = np.sqrt(2.0 * p * np.log(n))
        return self

    def _detect(self, x, t):
        return optim_max_cusum(x, t)
