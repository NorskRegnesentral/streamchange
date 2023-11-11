import abc
import numpy as np
import pandas as pd
from numbers import Number
from typing import Union, Tuple
from numba import njit

from ..penalties import BasePenalty, ConstantPenalty, BIC


class BaseOfflineCost:
    def fit(self, x: pd.DataFrame) -> "BaseOfflineCost":
        return self

    @abc.abstractmethod
    def __call__(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Calculate the optimal cost"""


@njit
def offline_l2_cost(
    sums: np.ndarray,
    sums2: np.ndarray,
    weights: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    if len(ends) == 1:
        ends = np.repeat(ends, len(starts))
    partial_sums = sums[ends + 1] - sums[starts]
    partial_sums2 = sums2[ends + 1] - sums2[starts]
    weights = weights[ends - starts + 1]
    costs = np.sum(partial_sums2 - partial_sums**2 / weights, axis=1)
    return costs


class OfflineL2Cost(BaseOfflineCost):
    def __init__(self):
        self.sums = None
        self.sums2 = None

    def fit(self, x):
        self.n = x.shape[0]
        self.p = x.shape[1]

        # 0.0 as first row to make calculations work also for start = 0
        self.sums = np.zeros((self.n + 1, self.p))
        self.sums[1:] = np.cumsum(x.values, axis=0)
        self.sums2 = np.zeros((self.n + 1, self.p))
        self.sums2[1:] = np.cumsum(x.values**2, axis=0)

        self.weights = np.tile(np.arange(0, self.n + 1).reshape(-1, 1), (1, self.p))

    def check_is_fitted(self):
        if self.sums is None:
            raise RuntimeError("OfflineL2Cost must be fit before calling.")

    def __call__(self, starts, ends):
        self.check_is_fitted()
        if isinstance(ends, Number):
            ends = np.array([ends])
        return offline_l2_cost(self.sums, self.sums2, self.weights, starts, ends)
