import abc
from numbers import Number
from typing import Tuple, Union, Callable
import copy
import pandas as pd
import numpy as np

from ..penalties import BasePenalty, ConstantPenalty


class BaseScore:
    @abc.abstractmethod
    def reset(self) -> "BaseScore":
        return self

    @property
    @abc.abstractmethod
    def value(self) -> float:
        pass

    @abc.abstractmethod
    def update(self, x: Union[Number, np.ndarray]) -> "BaseScore":
        """Update the score with a single data point.

        Parameters
        ----------
        x
            One observation row-vector.

        Returns
        -------
        self
        """
        return self

    def fit(self, x: pd.DataFrame) -> "BaseScore":
        self.reset()
        x = x.dropna()
        self.values_ = []
        for x_t in x.values:
            self.update(x_t)
            self.values_.append(self.value)
        self.values_ = pd.Series(self.values_, index=x.index)
        return self


class BaseRawScore(BaseScore):
    """A raw changepoint score. It is always >= 0"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._score = 0.0
        return self

    @property
    def value(self):
        return float(self._score)

    def penalise(self, penalty: BasePenalty):
        return PenalisedScore(self, penalty)


class BasePenalisedScore(BaseScore):
    """A penalised changepoint score where self.value > 0 means a change is detected."""

    def __init__(self, penalty: Tuple[BasePenalty, Number]):
        if isinstance(penalty, Number):
            penalty = ConstantPenalty(penalty)
        self.penalty = penalty
        self.reset()

    def reset(self):
        self._penalised_score = -self.penalty()
        return self

    @property
    def value(self):
        """Get the current value of the penalised score."""
        return float(self._penalised_score)


class PenalisedScore(BasePenalisedScore):
    def __init__(self, score: BaseRawScore, penalty: BasePenalty):
        self.score = score
        super().__init__(penalty)
        self.reset()

    def reset(self):
        self.score.reset()
        super().reset()
        return self

    def update(self, x):
        self.score.update(x)
        self._penalised_score = self.score.value - self.penalty()
        return self


class AggregatedScore(BaseRawScore):
    def __init__(self, base_score: BaseRawScore, aggregator: Callable = np.sum):
        self.base_score = base_score
        self.aggregator = aggregator
        self.reset()

    def reset(self):
        self.scores = None
        super().reset()
        return self

    def _init_scores(self, x):
        self.scores = [copy.deepcopy(self.base_score) for _ in x]

    def update(self, x):
        if not isinstance(x, np.ndarray):
            # Assumes x to be a number if it is not an array.
            x = np.asarray([x])

        if self.scores is None:
            self._init_scores(x)

        for i, x_i in enumerate(x):
            self.scores[i].update(x_i)

        self._score = self.aggregator([score.value for score in self.scores])
        return self


class LordenPollakScore(BaseRawScore):
    def __init__(self, rho: Number):
        self.rho = rho
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        super().reset()
        return self

    def update(self, x: Number):
        mean = self.sum / self.n if self.n > 0 else 0
        mu = max(mean, self.rho)
        self._score = max(0, self._score + mu * x - mu**2 / 2)
        if self._score < 1e-8:
            self.reset()
        else:
            self.n += 1
            self.sum += x
        return self
