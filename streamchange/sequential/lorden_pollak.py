from streamchange.base import ChangeDetector
from numbers import Number
from typing import Tuple
import copy
import pandas as pd
import numpy as np

from ..penalties import BasePenalty, ConstantPenalty


class LordenPollakCUSUM(ChangeDetector):
    def __init__(
        self,
        rho: Number,
        penalty: Tuple[BasePenalty, Number],
        reset_on_change: bool = True,
        restart_delay: int = 0,
    ):
        if isinstance(penalty, Number):
            penalty = ConstantPenalty(penalty)
        self.penalty = penalty
        self.rho = rho
        self.reset_on_change = reset_on_change
        self.restart_delay = restart_delay if reset_on_change else 0
        self.restart_counter = 0
        self.reset()

    def reset(self):
        super().reset()
        self.n = 0
        self.sum = 0.0
        self.score = 0.0

    def get_penalty(self):
        return self.penalty

    def update(self, x: Number):
        if self.reset_on_change and self.change_detected:
            self.reset()

        super().reset()
        if self.restart_counter < self.restart_delay:
            self.restart_counter += 1
            return self

        mean = self.sum / self.n if self.n > 0 else 0
        mu = max(mean, self.rho)
        self.score = max(0, self.score + mu * x - mu**2 / 2)
        if self.score > self.penalty():
            self._changepoints = [self.n + 1]
            self.restart_counter = 0

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

        self.scores_ = pd.Series(np.zeros(len(x)), index=times)
        alarm_indices = []
        cpts = []
        for t, x_t in x.items():
            self.update(x_t)
            self.scores_[t] = self.score
            if self.change_detected:
                alarm_indices.append(t)
                cpts.append(t - self.changepoints[0])  # Can only contain one cpt.
        self.alarm_times_ = times[alarm_indices].tolist()
        self.changepoints_ = times[cpts].tolist()
        return self

    def predict(self, x: pd.DataFrame = None) -> list:
        if x is None:
            return copy.deepcopy(self.changepoints_)
        else:
            # TODO: Complete
            raise RuntimeError(
                "Prediction for new observations is not implemented yet."
            )
