import numbers

from .change_detector import ChangeDetector


class LordenPollakCUSUM(ChangeDetector):
    def __init__(self, rho: numbers.Number, threshold: numbers.Number = 0):
        self.rho = rho
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._changepoints = []
        self.n = 0
        self.sum = 0.0
        self.score = 0.0

    def update(self, x: numbers.Number):
        self._changepoints = []

        mean = self.sum / self.n if self.n > 0 else 0
        mu = max(mean, self.rho)
        self.score = max(0, self.score + mu * x - mu**2 / 2)
        if self.score > self.threshold:
            self._changepoints = [-self.n - 1]

        if self.score < 1e-8:
            self.reset()
        else:
            self.n += 1
            self.sum += x
        return self
