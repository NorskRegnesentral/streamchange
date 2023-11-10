import numpy as np
import pandas as pd
from math import floor
from numbers import Number
from typing import Tuple
import copy

from streamchange.base import NumpyDeque
from .costs import BaseOfflineCost, OfflineL2Cost
from ..penalties import BasePenalty, BIC


class OfflinePelt:
    def __init__(
        self,
        cost: BaseOfflineCost = OfflineL2Cost(),
        penalty: BasePenalty = BIC(),
        minsl: int = 2,
    ):
        assert minsl >= 1
        self.minsl = minsl
        self.cost = cost
        self.penalty = penalty

    def fit(self, x: pd.DataFrame) -> "OfflinePelt":
        x = x.to_frame() if isinstance(x, pd.Series) else x

        self.cost.fit(x)
        pen = self.penalty()

        admissible = np.array([0])
        opt_cost = np.zeros(len(x) + 1)
        opt_cost[: self.minsl - 1] = -pen

        # Store the previous changepoint for each t.
        # Used to get the final set of changepoints after the loop.
        prev_cpts = [-1] * (self.minsl - 1)

        for t in range(self.minsl - 1, len(x)):
            admissible = np.concatenate((admissible, [t - self.minsl + 1]))

            admissible_opt_costs = opt_cost[admissible] + self.cost(admissible, t) + pen
            admissible_argmin = np.argmin(admissible_opt_costs)
            opt_cost[t] = admissible_opt_costs[admissible_argmin]
            prev_cpts.append(admissible[admissible_argmin] - 1)

            # trimming the admissible set
            admissible = admissible[admissible_opt_costs - pen <= opt_cost[t]]

        self.segments_ = self.get_segments(prev_cpts)
        self.changepoints_ = self.segments_["end"].tolist()
        return self

    @staticmethod
    def get_segments(prev_cpts: list) -> pd.DataFrame:
        segments = []
        i = len(prev_cpts) - 1
        while i >= 0:
            cpt_i = prev_cpts[i]
            segments.append(
                {
                    "start": cpt_i + 1,
                    "end": i,
                    "size": i - cpt_i,
                }
            )
            i = cpt_i
        return pd.DataFrame(segments).sort_values("start")
