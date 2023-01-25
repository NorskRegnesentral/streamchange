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
        self._changepoints = []
        self.test.reset()
        self.window.reset()
        return self

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
