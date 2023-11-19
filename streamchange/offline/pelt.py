import abc
import numpy as np
import pandas as pd
from numba import njit, int64, float64
from numba.experimental import jitclass

from ..penalties import BasePenalty, BIC
from .utils import colcumsum


class BasePeltCost:
    def fit(self, x: pd.DataFrame) -> "BasePeltCost":
        return self

    @abc.abstractmethod
    def get(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Calculate the optimal cost"""


@jitclass([("sums", float64[:, :]), ("sums2", float64[:, :]), ("weights", int64[:, :])])
class L2Cost(BasePeltCost):
    n: int
    p: int

    def __init__(self):
        pass

    def fit(self, x: np.ndarray):
        self.n = x.shape[0]
        self.p = x.shape[1]

        # 0.0 as first row to make calculations work also for start = 0
        self.sums = np.zeros((self.n + 1, self.p))
        self.sums[1:] = colcumsum(x)
        self.sums2 = np.zeros((self.n + 1, self.p))
        self.sums2[1:] = colcumsum(x**2)

        self.weights = np.arange(0, self.n + 1).repeat(self.p).reshape(-1, self.p)

    def check_is_fitted(self):
        if self.sums is None:
            raise RuntimeError("OfflineL2Cost must be fit before calling.")

    def get(self, starts: np.ndarray, ends: np.ndarray):
        self.check_is_fitted()

        if len(ends) == 1:
            ends = np.repeat(ends, len(starts))

        partial_sums = self.sums[ends + 1] - self.sums[starts]
        partial_sums2 = self.sums2[ends + 1] - self.sums2[starts]
        weights = self.weights[ends - starts + 1]
        costs = np.sum(partial_sums2 - partial_sums**2 / weights, axis=1)
        return costs


@njit
def fit_pelt(x: np.ndarray, cost, penalty: float, minsl: int = 2) -> list:
    cost.fit(x)
    admissible = np.array([0])
    opt_cost = np.zeros(len(x) + 1)
    opt_cost[: minsl - 1] = -penalty

    # Store the previous changepoint for each t.
    # Used to get the final set of changepoints after the loop.
    prev_cpts = [-1] * (minsl - 1)

    for t in range(minsl - 1, len(x)):
        t = np.array([t])
        admissible = np.concatenate((admissible, t - minsl + 1))

        costs = cost.get(admissible, t)
        admissible_opt_costs = opt_cost[admissible] + costs + penalty
        admissible_argmin = np.argmin(admissible_opt_costs)
        opt_cost[t] = admissible_opt_costs[admissible_argmin]
        prev_cpts.append(admissible[admissible_argmin] - 1)

        # trimming the admissible set
        admissible = admissible[admissible_opt_costs - penalty <= opt_cost[t]]

    return prev_cpts


class Pelt:
    def __init__(
        self,
        cost: BasePeltCost = L2Cost(),
        penalty: BasePenalty = BIC(),
        minsl: int = 2,
    ):
        assert minsl >= 1
        self.minsl = minsl
        self.cost = cost
        self.penalty = penalty

    def fit(self, x: pd.DataFrame):
        x = x.to_numpy()
        prev_cpts = fit_pelt(x, self.cost, self.penalty(), self.minsl)
        self.segments_ = self.get_segments(prev_cpts)
        self.changepoints_ = self.segments_["end"].tolist()
        self.n_detections_ = len(self.changepoints_)
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
        return pd.DataFrame(segments).sort_values("start").reset_index(drop=True)
