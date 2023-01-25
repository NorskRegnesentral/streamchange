import numbers
import numpy as np
from numba import njit

from .amoc_test import AMOCTest


@njit
def univariate_cusum_transform(x: np.ndarray, t: np.ndarray):
    n = x.size
    sums = x.cumsum()
    cusum = np.sqrt(n / (t * (n - t))) * (t / n * sums[-1] - sums[t - 1])
    return cusum


@njit
def optim_univariate_cusum(x: np.ndarray, t: np.ndarray):
    n = x.size
    cusum = univariate_cusum_transform(x, t)
    abs_cusum = np.abs(cusum)
    argmax = abs_cusum.argmax()
    return abs_cusum[argmax], t[argmax] - n - 1


class UnivariateCUSUM(AMOCTest):
    def __init__(self, threshold: numbers.Number = 0.0, minsl: int = 1):
        super().__init__()
        assert threshold >= 0.0
        self.threshold = threshold
        self.minsl = minsl

    def set_default_threshold(self, n: float):
        self.threshold = np.sqrt(2.0 * np.log(n))
        return self

    def detect(self, x):
        t = np.arange(self.minsl, x.size - self.minsl + 1)
        if t.size > 0:
            self.test_stat, self._changepoint = optim_univariate_cusum(x, t)
        else:
            self.test_stat, self._changepoint = (0, None)
        self._change_detected = self.test_stat > self.threshold
        return self
