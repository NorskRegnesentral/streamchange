import pandas as pd
import numpy as np
import copy
from typing import Tuple, Union
from numba import njit

from .detector import SequentialChangeDetector
from ..tuners import BasePenaltyTuner
from ..penalties import ConstantPenalty


class SequentialScorePenaltyTuner(BasePenaltyTuner):
    def __init__(
        self,
        detector: SequentialChangeDetector,
        target_detections: int = 0,
        refit: bool = True,
        index_margin: Union[int, pd.Timestamp] = None,
        score_value_margin: float = None,
        score_quantile_margin: float = None,
    ):
        self.detector = detector
        self.target_detections = target_detections
        self.refit = refit

        not_none_count = sum(
            [
                index_margin is not None,
                score_value_margin is not None,
                score_quantile_margin is not None,
            ]
        )
        if not_none_count > 1:
            raise ValueError(
                "Only one of index_margin, score_value_margin, score_quantile_margin can be specified."
            )
        if not_none_count == 0:
            raise ValueError(
                "One of index_margin, score_value_margin, score_quantile_margin must be specified."
            )
        self.index_margin = index_margin
        self.score_value_margin = score_value_margin
        self.score_quantile_margin = score_quantile_margin

    def _get_event_boundaries(
        self,
        scores: pd.Series,
        score_argmax: Union[int, pd.Timestamp],
    ):
        if self.index_margin is not None:
            lower = scores.index[score_argmax] - self.index_margin
            upper = scores.index[score_argmax] + self.index_margin
        else:
            if (
                self.score_quantile_margin is not None
                and self.score_value_margin is None
            ):
                # On first call, calculate the score quantile.
                self.score_value_margin = scores.quantile(self.score_quantile_margin)

            left_scores = scores.iloc[:score_argmax]
            left_scores = left_scores.loc[left_scores <= self.score_value_margin]
            lower = scores.index[0] if len(left_scores) == 0 else left_scores.index[-1]
            right_scores = scores.loc[score_argmax + 1 :]
            right_scores = right_scores.loc[right_scores <= self.score_value_margin]
            upper = (
                scores.index[-1] if len(right_scores) == 0 else right_scores.index[0]
            )

        return lower, upper

    def fit(self, x: pd.DataFrame) -> "SequentialScorePenaltyTuner":
        # The penalty tuner is based on not resetting the detector on change
        # as well as running the detector with 0 penalty to assess the raw score.
        detector = copy.deepcopy(self.detector)
        detector.reset_on_change = False
        detector.penalised_score.penalty = ConstantPenalty(0.0)

        # Run the detector on the data to get the raw scores.
        detector.fit(x)
        scores = copy.deepcopy(detector.penalised_scores_)

        self.penalties = []
        detection_counts = list(range(self.target_detections + 1))
        for detection_count in detection_counts:
            score_argmax = scores.argmax()
            self.penalties.append(scores.iloc[score_argmax])
            lower, upper = self._get_event_boundaries(scores, score_argmax)
            scores.loc[lower:upper] = 0.0

        self.summary = pd.DataFrame(self._summarise())
        penalty_scale_ = self.summary["penalty_scale"].iloc[-1]
        self.detector_ = copy.deepcopy(self.detector)
        self.detector_.get_penalty().scale = penalty_scale_
        if self.refit:
            self.detector_.fit(x)

        return self

    def _summarise(self) -> dict:
        default_penalty = self.detector.get_penalty().default_penalty()
        penalties = np.array(self.penalties)
        return {
            "detection_count": np.arange(self.target_detections + 1),
            "penalty": penalties,
            "penalty_scale": penalties / default_penalty,
        }
