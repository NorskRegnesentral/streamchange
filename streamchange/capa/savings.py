import abc
import numpy as np
from numbers import Number
from typing import Union
from numba import njit


class BaseSaving:
    def __init__(self, penalty: float = None, arl: int = 10000, p: int = 1):
        self.arl = arl
        self.p = p
        self.penalty = self.default_penalty(arl, p) if penalty is None else penalty

    @staticmethod
    def default_penalty(n: int, p: int = 1) -> float:
        """Default penalty as function of n and p"""
        phi = np.log(n)
        return p + 2 * np.sqrt(p * phi) + 2 * phi

    @abc.abstractmethod
    def opt(self, x: Union[Number, np.ndarray]) -> Number:
        """Calculate the optimal saving"""

    @abc.abstractmethod
    def cumopt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the optimal saving cumulatively from the _right_"""


@njit
def cumopt_constmeanl2(x: np.ndarray, penalty: float) -> np.ndarray:
    sums = np.cumsum(x[::-1])[::-1]
    k = np.arange(x.shape[0], 0, -1)
    return sums**2 / k - penalty


class ConstMeanL2(BaseSaving):
    def __init__(self, penalty=None, arl=10000, p=1):
        super().__init__(penalty, arl, p)

    def opt(self, x):
        if isinstance(x, Number):
            return x**2 - self.penalty
        else:
            return np.sum(x) ** 2 / x.size - self.penalty

    def cumopt(self, x):
        return cumopt_constmeanl2(x, self.penalty)
