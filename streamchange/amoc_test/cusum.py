import abc
import numbers
import numpy as np
from numba import njit

from .amoc_test import AMOCTest


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
    return agg_cusum[argmax], t[argmax] - n - 1


class CUSUM(AMOCTest):
    def __init__(self, threshold: numbers.Number = 0.0, minsl: int = 1):
        super().__init__()
        assert threshold >= 0.0
        self.threshold = threshold
        self.minsl = minsl

    @abc.abstractmethod
    def _detect(self, x, t):
        pass

    def detect(self, x):
        if x.shape[0] >= 2 * self.minsl:
            t = np.arange(self.minsl, x.shape[0] - self.minsl + 1)
            self.test_stat, self._changepoint = self._detect(x, t)
            self._change_detected = self.test_stat > self.threshold
        else:
            self.reset()
        return self


# Univariate CUSUM
# ------------------------------------------------------------------------------
@njit
def optim_univariate_cusum(x: np.ndarray, t: np.ndarray):
    cusum = univariate_cusum_transform(x, t)
    abs_cusum = np.abs(cusum)
    return _optim(abs_cusum, t, x.shape[0])


class UnivariateCUSUM(CUSUM):
    def set_default_threshold(self, n: float):
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
