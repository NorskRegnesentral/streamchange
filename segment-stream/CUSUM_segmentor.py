from .segmentor import Segmentor

import numpy as np
from numba import njit


@njit
def univariate_cusum_test(sums: np.ndarray):
    n = sums.size
    t = np.arange(1, n)
    tests = np.sqrt(n / (t * (n - t))) * np.abs(t / n * sums[-1] - sums[:-1])
    cpt = tests.argmax() + 1
    cpt_test = tests[cpt - 1]
    return cpt_test, cpt


@njit
def univariate_cusum_tests_for(starts: np.ndarray, ends: np.ndarray, x: np.ndarray):
    # starts and ends input as np.ndarrays is much faster than other options in numba.
    sums = np.concatenate((np.array([0.0]), x.cumsum()))
    tests = []
    cpts = []
    for start, end in zip(starts, ends):
        current_sums = sums[start + 1 : end + 1] - sums[start]
        test, cpt = univariate_cusum_test(current_sums)
        tests.append(test)
        cpts.append(start + cpt)

    return tests, cpts


class CusumSegmentor(Segmentor):
    def set_default_penalty(self, n: float):
        self.penalty = np.sqrt(2.0 * np.log(n))

    def test(self, x: np.ndarray) -> np.ndarray:
        """
        Computes CUSUM change-point tests for all possible change-point locations in x.

        Arguments:
            x (np.ndarray): A vector of length n of numerical values to look for change-points in.

        Returns:
            (Change-point test statistic, most likely change-point)
        """
        if x.size < self.min_size_window:
            return 0.0, 0
        return univariate_cusum_test(x.cumsum())

    def tests_for(self, starts: list, ends: list, x: np.ndarray):
        return univariate_cusum_tests_for(np.array(starts), np.array(ends), x)
