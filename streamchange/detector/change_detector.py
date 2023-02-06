import abc
from typing import Union
import numbers


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

        Changepoints are stored as their negative index within the current window.
        This makes it easy to extract changepoints also outside this class,
        where the relevant temporal frame of reference is.
        """
        return self._changepoints

    @abc.abstractmethod
    def update(self, x: Union[numbers.Number, dict]) -> "ChangeDetector":
        """Update the change detector with a single data point.

        Parameters
        ----------
        x
            One observation row-vector.

        Returns
        -------
        self
        """
