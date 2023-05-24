import abc
import numpy as np
from numbers import Number
from typing import Union, Tuple
from numba import njit

from ..penalties import BasePenalty, ConstantPenalty, ChiSquarePenalty


class BaseSaving:
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
def cumopt_constmeanl2(x: np.ndarray) -> np.ndarray:
    sums = np.cumsum(x)
    k = np.arange(1, x.shape[0] + 1)
    return sums**2 / k


class ConstMeanL2(BaseSaving):
    def __init__(self, penalty=ChiSquarePenalty()):
        super().__init__(penalty)

    def opt(self, x):
        if isinstance(x, Number):
            return x**2 - self.penalty()
        else:
            return np.sum(x) ** 2 / x.size - self.penalty()

    def cumopt(self, x):
        return cumopt_constmeanl2(x) - self.penalty()
