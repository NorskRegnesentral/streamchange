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
def nb_sum(x: np.ndarray):
    return np.sum(x)


@njit
def nb_max(x: np.ndarray):
    return np.max(x)


@njit
def fit_cusum0_score(x: np.ndarray, window_sizes: np.ndarray, agg):
    x_original_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    p = x.shape[1]

    sums = np.zeros((n + 1, p))
    sums[1:] = colcumsum(x)
    weights = 1 / window_sizes.reshape(-1, 1)

    scores = np.zeros(n)
    for i in range(1, n + 1):
        before_window_sums = sums[np.maximum(0, i - window_sizes)]
        partial_sums = sums[i] - before_window_sums
        cusums = weights * partial_sums**2
        max_cusums = colmax(cusums)
        score = agg(max_cusums)
        scores[i - 1] = score

    return scores


@njit
def fit_cusum0_detector(x: np.ndarray, penalty: float, window_sizes: np.ndarray, agg):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = x.shape[0]
    p = x.shape[1]

    sums = np.zeros((n + 1, p))
    sums[1:] = colcumsum(x)
    weights = 1 / window_sizes.reshape(-1, 1)

    scores = np.zeros(n)
    alarms = [-1]
    for i in range(1, n + 1):
        before_window_sums = sums[np.maximum(alarms[-1] + 1, i - window_sizes)]
        partial_sums = sums[i] - before_window_sums
        cusums = weights * partial_sums**2
        max_cusums = colmax(cusums)
        score = np.sum(max_cusums)
        scores[i - 1] = score

        if score > penalty:
            alarms.append(i - 1)

    return alarms[1:], scores
