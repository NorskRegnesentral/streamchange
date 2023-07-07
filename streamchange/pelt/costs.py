import abc
import numpy as np
from numbers import Number
from typing import Union, Tuple
from numba import njit

from ..penalties import BasePenalty, ConstantPenalty, BIC


class BaseCost:
    def __init__(self, penalty: Tuple[BasePenalty, Number]):
        if isinstance(penalty, Number):
            penalty = ConstantPenalty(penalty)
        self.penalty = penalty

    @abc.abstractmethod
    def opt(self, x: Union[Number, np.ndarray]) -> Number:
        """Calculate the optimal saving"""

    @abc.abstractmethod
    def cumopt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the optimal saving cumulatively from the _right_"""


@njit
def cumopt_l2_cost(x: np.ndarray) -> np.ndarray:
    # TODO: Extend to multiple dimensions
    sums = np.cumsum(x)
    sums2 = np.cumsum(x**2)
    costs = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        # costs[i] = np.sum(i * (sums2[i] / i - (sums[i] / i) ** 2))
        costs[i] = sums2[i] - sums[i] ** 2 / (i + 1)
    return costs


class L2Cost(BaseCost):
    def __init__(self, penalty=BIC()):
        super().__init__(penalty)

    def opt(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if isinstance(x, Number):
            return self.penalty()
        else:
            return x.shape[0] * x.var(axis=0).sum() + self.penalty()

    def cumopt(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        return cumopt_l2_cost(x) + self.penalty()
