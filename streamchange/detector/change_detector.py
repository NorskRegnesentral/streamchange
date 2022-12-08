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
    def update(self, x: dict) -> "ChangeDetector":
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            One observation row-vector.

        Returns
        -------
        self
        """
