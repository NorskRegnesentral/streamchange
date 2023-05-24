from streamchange.base import ChangeDetector
from numbers import Number
from typing import Tuple

from ..penalties import BasePenalty, ConstantPenalty


class LordenPollakCUSUM(ChangeDetector):
    def __init__(self, rho: Number, penalty: Tuple[BasePenalty, Number]):
        if isinstance(penalty, Number):
            penalty = ConstantPenalty(penalty)
        self.penalty = penalty
        self.rho = rho
        self.reset()

    def reset(self):
        super().reset()
        self.n = 0
        self.sum = 0.0
        self.score = 0.0

    def update(self, x: Number):
        super().reset()

        mean = self.sum / self.n if self.n > 0 else 0
        mu = max(mean, self.rho)
        self.score = max(0, self.score + mu * x - mu**2 / 2)
        if self.score > self.penalty():
            self._changepoints = [self.n + 1]

        if self.score < 1e-8:
            self.reset()
        else:
            self.n += 1
            self.sum += x
        return self
