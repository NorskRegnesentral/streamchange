import abc
import numpy as np
from numbers import Number
from typing import Union


class BaseSaving:
    def __init__(self, penalty: float = None, p: int = 1):
        if penalty is None:
            arl = 10000
            self.set_penalty(arl, p=p)
        else:
            self.penalty = penalty

    @abc.abstractmethod
    def set_penalty(self, n: int, p: int = 1):
        return self

    @abc.abstractmethod
    def opt(self, x: np.ndarray) -> Number:
        """Calculate the optimal saving"""

    @abc.abstractmethod
    def cumopt(self, x: np.ndarray) -> np.ndarray:
        """Calculate the optimal saving cumulatively from the _right_"""


class ConstMeanL2(BaseSaving):
    def __init__(self, penalty: float = None, p: int = 1):
        super().__init__(penalty, p)

    def set_penalty(self, n: int, p: int = 1):
        phi = np.log(n)
        self.penalty = p + 2 * np.sqrt(p * phi) + 2 * phi
        return self

    def opt(self, x: Union[Number, np.ndarray]):
        if isinstance(x, Number):
            return x**2 - self.penalty
        else:
            return np.sum(x) ** 2 / x.size - self.penalty

    def cumopt(self, x):
        sums = np.cumsum(x[::-1])[::-1]
        k = np.arange(x.shape[0], 0, -1)
        return sums**2 / k - self.penalty
