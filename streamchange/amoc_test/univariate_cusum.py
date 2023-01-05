import numpy as np
from numba import njit

from .amoc_test import AMOCTest


@njit
def univariate_cusum_transform(x: np.ndarray):
    n = x.size
    t = np.arange(1, n)
    sums = x.cumsum()
    cusum = np.sqrt(n / (t * (n - t))) * (t / n * sums[-1] - sums[:-1])
    return cusum


@njit
def optim_univariate_cusum(x: np.ndarray):
    abs_cusum = np.abs(univariate_cusum_transform(x))
    n = x.size
    argmax_cusum = abs_cusum.argmax()
    max_cusum = abs_cusum[argmax_cusum]
    return max_cusum, argmax_cusum - n


# @njit
# def optim_univariate_cusum_for(starts: np.ndarray, ends: np.ndarray, x: np.ndarray):
#     # starts and ends input as np.ndarrays is much faster than other options in numba.
#     sums = np.concatenate((np.array([0.0]), x.cumsum()))
#     tests = []
#     cpts = []
#     for start, end in zip(starts, ends):
#         current_sums = sums[start + 1 : end + 1] - sums[start]
#         test, cpt = optim_univariate_cusum(current_sums)
#         tests.append(test)
#         cpts.append(start + cpt)

#     return tests, cpts


class UnivariateCUSUM(AMOCTest):
    def __init__(self, threshold=0.0):
        super().__init__()
        assert threshold >= 0.0
        self.threshold = threshold

    def set_default_threshold(self, n: float):
        self.threshold = np.sqrt(2.0 * np.log(n))
        return self

    def detect(self, x):
        self.test_stat, self._changepoint = optim_univariate_cusum(x)
        self._change_detected = self.test_stat > self.threshold
        return self

    # def optim_for(self, starts: list, ends: list, x: np.ndarray):
    #     return optim_univariate_cusum_for(np.array(starts), np.array(ends), x)
