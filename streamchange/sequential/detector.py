from numbers import Number
from typing import Union
import copy
import pandas as pd
import numpy as np

from .scores import BasePenalisedScore


class SequentialChangeDetector:
    def __init__(
        self,
        penalised_score: BasePenalisedScore,
        reset_on_change: bool = True,
        restart_delay: int = 0,
    ):
        self.penalised_score = penalised_score
        self.reset_on_change = reset_on_change
        self.restart_delay = restart_delay if reset_on_change else 0
        self.reset()

    def reset(self):
        self.penalised_score.reset()
        self.restart_counter = 0
        return self

    @property
    def change_detected(self) -> bool:
        return self.penalised_score.value > 0

    @property
    def changepoint(self) -> int:
        return self.penalised_score.changepoint

    def get_penalty(self):
        return self.penalised_score.penalty

    def update(self, x: Union[Number, np.ndarray]):
        if self.reset_on_change and self.change_detected:
            self.restart_counter = 0
            self.reset()

        if self.restart_counter < self.restart_delay:
            self.restart_counter += 1
            return self

        self.penalised_score.update(x)
        return self

    def fit(self, x: pd.DataFrame):
        self.reset()
        x = x.dropna()
        times = x.index

        penalised_scores_ = []
        self.alarms_ = []
        self.changepoints_ = []
        for t, x_t in zip(x.index, x.values):
            self.update(x_t)
            penalised_scores_.append(self.penalised_score.value)
            if self.change_detected:
                self.alarms_.append(t)
                if self.changepoint:
                    self.changepoints_.append(self.changepoint)
        self.penalised_scores_ = pd.Series(penalised_scores_, index=times)
        return self

    def predict(self, x: pd.DataFrame = None) -> list:
        if x is None:
            return copy.deepcopy(self.alarms_)
        else:
            raise RuntimeError("Prediction for new observations is not supported.")

    def fit_predict(self, x: pd.DataFrame) -> list:
        return self.fit(x).predict()

    def transform(self, x: pd.DataFrame) -> pd.Series:
        return self.fit(x).penalised_scores_
