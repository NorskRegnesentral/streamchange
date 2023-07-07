import numpy as np
import pandas as pd
from numbers import Number
from typing import Tuple
import copy

from streamchange.base import NumpyDeque
from .costs import BaseCost, L2Cost


class Pelt:
    # Assumes x is standardised outside of this class.
    def __init__(
        self,
        cost: BaseCost = L2Cost(),
        minsl=1,
        maxsl=1000,
    ):
        assert minsl >= 1
        assert maxsl > minsl
        self.minsl = minsl
        self.maxsl = maxsl
        self.cost = cost
        self.reset()

    def reset(self):
        self.window = NumpyDeque(self.maxsl)
        self.opt_cost = NumpyDeque(self.maxsl)
        self.opt_cost.appendleft(-self.cost.penalty())
        self.last_cpt = 0
        return self

    def get_penalty(self):
        return self.cost.penalty

    @property
    def change_detected(self):
        return self.last_cpt > 0

    def update(self, x: Number):
        self.window.appendleft(x)
        n = len(self.window)
        if n >= self.minsl:
            opt_costs = self.opt_cost.values[self.minsl - 1 :]
            costs = self.cost.cumopt(self.window.values)[self.minsl - 1 :]
            candidate_costs = opt_costs + costs
            opt_candidate = np.argmin(candidate_costs)
            self.last_cpt = self.minsl + opt_candidate
            self.opt_cost.appendleft(candidate_costs[opt_candidate])
        else:
            cost = self.cost.opt(self.window.values)
            self.opt_cost.appendleft(self.opt_cost.values[0] + cost)

        # TODO: Add pruning.
        return self

    @staticmethod
    def extract_segments(last_cpts: pd.Series) -> list:
        times = last_cpts.index
        segments = []
        i = -1
        while i >= -last_cpts.size:
            cpt_i = last_cpts.values[i]
            segments.append(
                {
                    "start": times[i - cpt_i + 1],
                    "end": times[i],
                    "size": abs(cpt_i),
                }
            )
            i -= cpt_i
        return segments

    def fit(self, x: pd.Series) -> "Pelt":
        self.reset()
        x = x.dropna()
        last_cpts = []
        for value in x.values:
            self.update(value)
            last_cpts.append(self.last_cpt)
        last_cpts = pd.Series(last_cpts, index=x.index, dtype=int)
        self.segments_ = self.extract_segments(last_cpts)
        self.changepoints_ = [s["end"] for s in self.segments_]
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "changepoints_"):
            msg = f"This instance of {type(self).__name__} is not fitted yet."
            raise RuntimeError(msg)

    def predict(self, x: pd.Series = None) -> list:
        self._check_is_fitted()
        if x is None:
            return copy.deepcopy(self.segments_)
        else:
            # TODO: Complete
            raise RuntimeError("Prediction for new observation is not implemented yet.")

    def fit_predict(self, x: pd.DataFrame) -> list:
        return self.fit(x).predict()
