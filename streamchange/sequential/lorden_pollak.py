from streamchange.base import ChangeDetector
from numbers import Number
from typing import Tuple
import copy
import pandas as pd

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

    def fit(self, x: pd.Series) -> "LordenPollakCUSUM":
        self.reset()
        x = x.dropna()
        times = x.index
        x = x.reset_index(drop=True)
        cpts = []
        for t, x_t in x.items():
            self.update(x_t)
            if self.change_detected:
                cpts += [t - cpt for cpt in self.changepoints]
                self.reset()
        self.changepoints_ = times[cpts].tolist()
        return self

    def predict(self, x: pd.DataFrame = None) -> list:
        if x is None:
            return copy.deepcopy(self.changepoints_)
        else:
            # TODO: Complete
            raise RuntimeError("Prediction for new observation is not implemented yet.")

    def fit_predict(self, x: pd.DataFrame) -> list:
        return self.fit(x).predict()
