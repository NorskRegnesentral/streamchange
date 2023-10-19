import pandas as pd
import numpy as np
from typing import Tuple, Callable
from numba import njit

from .window_segmentor import WindowSegmentor
from ..tuners import BasePenaltyTuner


@njit
def make_random_intervals(
    n: int,
    min_window: int,
    max_window: int,
    prob: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    ends_range = range(0, n - min_window)
    starts_range = range(min_window, max_window + 1)
    for end in ends_range:
        for start in starts_range:
            if np.random.uniform(0.0, 1.0) <= prob:
                ends.append(end)
                starts.append(end + start)
    starts = np.array(starts)
    ends = np.array(ends)
    starts = starts[starts <= n]
    ends = ends[starts <= n]
    return starts, ends


@njit
def make_stepwise_intervals(
    n: int,
    min_window: int,
    max_window: int,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    ends_range = range(0, n - min_window)
    starts_range = range(min_window, max_window + 1, step)
    for end in ends_range:
        for start in starts_range:
            ends.append(end)
            starts.append(end + start)
    starts = np.array(starts)
    ends = np.array(ends)
    starts = starts[starts <= n]
    ends = ends[starts <= n]
    return starts, ends


@njit
def make_dyadic_intervals(
    n: int,
    min_window: int,
    max_window: int,
    alpha: float = 1.5,
    step_proportion: int = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    interval_length = min_window
    while interval_length <= max_window:
        step = max(1, np.floor(step_proportion * interval_length))
        i = 0
        while i * step + interval_length <= n:
            ends.append(int(i * step))
            starts.append(int(i * step + interval_length))
            i += 1
        interval_length = max(interval_length + 1, np.floor(alpha * interval_length))
    return np.array(starts), np.array(ends)


def targetscaler(alpha=1.0):
    def selector(penalties):
        return max(alpha * penalties[-1], 1e-8)

    return selector


class RandomIntervalMaker:
    def __init__(self, prob: float):
        self.prob = prob

    def __call__(self, n: int, min_window: int, max_window: int):
        return make_random_intervals(n, min_window, max_window, self.prob)


class StepwiseIntervalMaker:
    def __init__(self, step: float):
        self.step = step

    def __call__(self, n: int, min_window: int, max_window: int):
        return make_stepwise_intervals(n, min_window, max_window, self.step)


class DyadicIntervalMaker:
    def __init__(self, alpha: float = 1.5, step_proportion: int = 3):
        self.alpha = alpha
        self.step_proportion = step_proportion

    def __call__(self, n: int, min_window: int, max_window: int):
        return make_dyadic_intervals(
            n, min_window, max_window, self.alpha, self.step_proportion
        )


class AMOCPenaltyTuner(BasePenaltyTuner):
    """
    Class for tuning the penalty of an AMOCEstimator in WindowSegmentor.

    Parameters
    ----------
    detector :
        WindowSegmentor to tune.

    target_detections:
        Target number of changepoints to be detected in the training data.

    """

    def __init__(
        self,
        detector: WindowSegmentor,
        target_detections: int = 1,
        interval_generator="dyadic",
        prob: float = 0.1,
        step: int = 5,
        alpha: float = 1.5,
        step_proportion: float = 0.25,
        selector: Callable = targetscaler(alpha=1.0),
    ):
        self.detector = detector
        self.target_detections = target_detections
        self.interval_generator = interval_generator
        self.prob = prob
        self.step = step
        self.alpha = alpha
        self.step_proportion = step_proportion
        self.selector = selector
        self._set_interval_maker()

    def _set_interval_maker(self):
        if self.interval_generator == "random":
            self._make_intervals = RandomIntervalMaker(self.prob)
        elif self.interval_generator == "stepwise":
            self._make_intervals = StepwiseIntervalMaker(self.step)
        elif self.interval_generator == "dyadic":
            self._make_intervals = DyadicIntervalMaker(self.alpha, self.step_proportion)
        else:
            permitted_generators = ["random", "stepwise", "dyadic"]
            permitted_str = ", ".join(permitted_generators)
            raise ValueError(f"interval_generator must be one of {permitted_str}")

    def _detect_in(self, starts: list, ends: list):
        """
        Outputs the optimal test statistic and changepoint of each interval given
        by start[i]:(end[i]) in x.
        """
        scores = []
        cpts = []
        for start, end in zip(starts, ends):
            candidate_cpts = self.detector.candidate_cpts
            candidate_cpts = candidate_cpts[candidate_cpts < start - end]
            self.detector.estimator.fit(self.x[end:start], candidate_cpts)
            scores.append(self.detector.estimator.score)
            cpts.append(end + self.detector.estimator.changepoint)
        return np.array(scores), np.array(cpts)

    def _find_penalties(self) -> np.ndarray:
        starts, ends = self._make_intervals(
            self.x.shape[0],
            self.detector.min_window,
            self.detector.max_window,
        )
        scores, cpts = self._detect_in(starts, ends)
        self.scores = scores
        self.cpts = cpts
        penalties = np.zeros(self.target_detections)
        i = 0
        while (i < self.target_detections) & np.any(scores > 0.0):
            argmax = scores.argmax()
            penalties[i] = scores[argmax]
            max_cpt = cpts[argmax]
            cpt_in_interval = (max_cpt >= ends) & (max_cpt < starts)
            scores[cpt_in_interval] = 0.0
            i += 1
        return penalties

    def fit(self, x: pd.DataFrame) -> "AMOCPenaltyTuner":
        if x.shape[0] < self.target_detections:
            raise ValueError("x must contain more rows than target_detections.")

        if not x.index.is_monotonic_increasing:
            x = x.sort_index()

        # Non-penalised scores are used to find suitable penalties.
        self.detector.estimator.penalty.scale = 0

        # The smaller index means more recent throughout streamchange, thus reverse.
        self.x = x.to_numpy()[::-1]
        self.penalties = self._find_penalties()
        penalty = self.selector(self.penalties)
        self.penalty_scale_ = penalty / self.detector.estimator.penalty.value
        self.detector.estimator.penalty.scale = self.penalty_scale_
        return self

    def _summarise(self):
        results = {
            "detection_count": np.arange(self.target_detections) + 1,
            "penalty": self.penalties,
            "penalty_scale": self.penalties / self.detector.estimator.penalty.value,
        }
        return results
