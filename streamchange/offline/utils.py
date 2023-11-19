from typing import Callable
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
