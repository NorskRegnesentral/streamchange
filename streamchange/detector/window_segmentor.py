from .change_detector import ChangeDetector
from .detection_window import DetectionWindow
from streamchange.amoc_test import AMOCTest


class WindowSegmentor(ChangeDetector):
    """
    Class for testing-based changepoint detection.

    Parameters
    ----------
    test
        The single changepoint test.
    window
        The window mechanism to use.
    """

    def __init__(
        self,
        test: AMOCTest,
        window: DetectionWindow,
    ):
        self.test = test
        self.window = window
        self.reset()

    def reset(self) -> "WindowSegmentor":
        self.window.reset()
        self._changepoints = []
        return self

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

    def update(self, x):
        self._changepoints = []
        self.window.append(x)
        self.window.init_iter()
        while not self.window.stop_iter():
            self.test.detect(self.window.next())
            if self.test.change_detected:
                self._changepoints.append(self.test.changepoint)
                self.window.detection_reaction(self.test)
        return self


# TODO:
# Implement classical sequential tests (entirely recursive).
# Possibility to adjust candidate change-points (minimum and maximum seglen).
