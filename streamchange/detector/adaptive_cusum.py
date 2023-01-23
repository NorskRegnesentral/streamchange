import numbers

from .change_detector import ChangeDetector


class LordenPollakCUSUM(ChangeDetector):
    def __init__(
        self, rho: numbers.Number, threshold: numbers.Number = 0, zero_tol: float = 1e-8
    ):
        self.rho = rho
        self.threshold = threshold
        self.zero_tol = zero_tol
        self.score = 0.0
        self.sum = 0.0
        self.n = 0

    def update(self, x):
        assert len(x) == 1
        x = list(x.values())[0]

        if self.score < self.zero_tol:
            self.n = 0
            self.sum = 0.0
        else:
            self.n += 1
            self.sum += x
        mean = max(self.sum / self.n if self.n > 0 else 0, self.rho)
        self.score = max(0, self.score + mean * x - mean**2 / 2)
        self._change_detected = self.score > self.threshold
        return self
