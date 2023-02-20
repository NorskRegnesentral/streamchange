from numbers import Number
from typing import Union
import numpy as np


class NumpyDeque:
    def __init__(self, max_length: int = 1e6):
        """
        Parameters
        ----------
        max_length:
            The maximum size of the NumpyDeque.
        """
        self.max_length = max_length
        self.reset()

    def reset(self) -> "NumpyDeque":
        self.columns = None
        self._w = None
        return self

    def _init_window(self, x):
        if isinstance(x, np.ndarray):
            self.columns = None
            self._w = np.empty((0, *x.shape[1:]))
        if isinstance(x, Number):
            self.columns = None
            self._w = np.empty(0)
        else:
            self.columns = list(x.keys())
            self._w = np.empty((0, len(self.columns)))

    def _to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, Number):
            return np.array([x] if self.ndim == 1 else [[x]])
        else:
            return np.array([[x[key] for key in self.columns]])

    def pop(self, n: int = 1) -> np.ndarray:
        self._w = self._w[:-n]
        return self._w[-n:]

    def popleft(self, n: int = 1) -> np.ndarray:
        self._w = self._w[n:]
        return self._w[:n]

    def append(self, x: Union[Number, np.ndarray, dict]):
        if self._w is None:
            self._init_window(x)

        x = self._to_numpy(x)
        self._w = np.concatenate((self._w, x))
        if len(self) > self.max_length:
            self.popleft()

    def appendleft(self, x: Union[Number, np.ndarray, dict]):
        if self._w is None:
            self._init_window(x)

        x = self._to_numpy(x)
        self._w = np.concatenate((x, self._w))
        if len(self) > self.max_length:
            self.pop()

    @property
    def values(self) -> np.ndarray:
        return self._w

    @property
    def ndim(self) -> tuple:
        return self._w.ndim

    @property
    def shape(self) -> tuple:
        return self._w.shape

    def __len__(self) -> int:
        return self._w.shape[0]
