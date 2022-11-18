import numbers
import abc
import numpy as np


class ChangeDetector:
    def __init__(self):
        self._change_detected = False

    def _reset(self):
        self._change_detected = False

    @property
    def change_detected(self):
        return self._change_detected

    @abc.abstractmethod
    def update(self, x: np.ndarray) -> "ChangeDetector":
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            One observation row-vector.

        Returns
        -------
        self
        """


class UnivariateChangeDetector(ChangeDetector):
    def update(self, x: numbers.Number):
        return super().update(np.array([[x]]))


class MultivariateChangeDetector(ChangeDetector):
    def _reset(self):
        super()._reset()
        self._variable_names = None

    def _init_variable_names(self, x: dict):
        self._variable_names = list(x.keys())

    def _dict_to_nprow(self, x: dict):
        p = len(self._variable_names)
        return np.array([x[name] for name in self._variable_names]).reshape(1, p)

    def update(self, x: dict):
        if self._variable_names is None:
            self._init_variable_names(x)

        return super().update(self._dict_to_nprow(x))
