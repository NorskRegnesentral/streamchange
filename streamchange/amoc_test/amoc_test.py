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
        """The most likely location of a single changepoint.

        Changepoints are consistently stored as their negative index within the 
        current window. This makes it easy to extract changepoints also outside 
        this class, where the relevant temporal frame of reference is.
        """
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
