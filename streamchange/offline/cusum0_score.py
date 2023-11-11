import numpy as np
from numba import njit


@njit
def colmax(x: np.ndarray):
    maximums = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        maximums[j] = np.max(x[:, j])
    return maximums


@njit
def colcumsum(x: np.ndarray):
    cumsum = np.zeros_like(x)
    for j in range(x.shape[1]):
        cumsum[:, j] = np.cumsum(x[:, j])
    return cumsum


@njit
def fit_cusum0(x: np.ndarray, window_sizes: np.ndarray):
    x_original_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    p = x.shape[1]

    sums = np.zeros((n + 1, p))
    sums[1:] = colcumsum(x)
    weights = 1 / window_sizes.reshape(-1, 1)

    scores = np.zeros_like(x)
    for i in range(1, n + 1):
        before_window_sums = sums[np.maximum(0, i - window_sizes)]
        partial_sums = sums[i] - before_window_sums
        cusums = weights * partial_sums**2
        scores[i - 1] = colmax(cusums)

    return scores.reshape(x_original_shape)
