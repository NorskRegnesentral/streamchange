import numbers
import numpy as np
from numba import njit

from .amoc_test import AMOCTest
from .univariate_cusum import univariate_cusum_transform


@njit
def cusum_transform(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    p = x.shape[1]
    cusum = np.zeros((n - 1, p))
    for j in range(p):
        cusum[:, j] = univariate_cusum_transform(x[:, j])
    return cusum


@njit
def optim_sum_cusum(x: np.ndarray):
    cusum = cusum_transform(x)
    sum_cusum = np.abs(cusum).sum(axis=1)
    argmax_cusum = sum_cusum.argmax()  # The last index of a segment.
    max_cusum = sum_cusum[argmax_cusum]
    return max_cusum, argmax_cusum


class MultivariateCUSUM(AMOCTest):
    def __init__(self, threshold=0.0):
        super().__init__()
        assert threshold >= 0.0
        self.threshold = threshold

    def set_default_threshold(self, n: float, p: float):
        self.threshold = np.sqrt(2.0 * p * np.log(n))

    def detect(self, x: np.ndarray) -> tuple:
        self.test_stat, self._changepoint = optim_sum_cusum(x)
        self._change_detected = self.test_stat > self.threshold
        return self