import numpy as np
import abc


class AMOCTest:
    def __init__(self):
        self._change_detected = False
        self._changepoint = None

    @property
    def change_detected(self):
        return self._change_detected

    @property
    def changepoint(self):
        return self._changepoint

    @abc.abstractmethod
    def detect(self, x: np.ndarray) -> "AMOCTest":
        """Detect whether there is at least one changepoint in a data vector.

        Parameters
        ----------
        x
            Input values.

        Returns
        -------
        self

        """
