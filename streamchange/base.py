import abc
import copy
from numbers import Number
from typing import Union
import numpy as np
import pandas as pd

from .penalties import BasePenalty


class ChangeDetector:
    def __init__(self):
        self.reset()

    def reset(self):
        self._changepoints = []

    @property
    def change_detected(self):
        return len(self._changepoints) > 0

    @property
    def changepoints(self):
        """List of detected changepoints per iteration (call to update).

        Changepoints are defined as the last index of a homogenous segment.
        A changepoint is stored relative to the time of detection by its
        index backwards in time. For example, if t is the index of the current
        observation, the changepoint in the external data set is given by t - changepoint.
        """
        return self._changepoints

    @abc.abstractmethod
    def get_penalty(self) -> BasePenalty:
        """Get the penalty function of the change detector.

        Useful for tuning the penalty function across all change detectors, as
        the penalty function can be nested inside other objects in the ChangeDetector.
        """

    @abc.abstractmethod
    def update(self, x: Union[Number, dict]) -> "ChangeDetector":
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            One observation row-vector.

        Returns
        -------
        self
        """
        self.reset()
        return self


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
        self._init_window(0)
        return self

    def _init_window(self, x):
        if isinstance(x, np.ndarray):
            self.columns = None
            self._w = np.empty((0, *x.shape[1:]))
        elif isinstance(x, Number):
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
        if len(self) == 0:
            self._init_window(x)

        x = self._to_numpy(x)
        self._w = np.concatenate((self._w, x))
        if len(self) > self.max_length:
            self.popleft()

    def appendleft(self, x: Union[Number, np.ndarray, dict]):
        if len(self) == 0:
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
