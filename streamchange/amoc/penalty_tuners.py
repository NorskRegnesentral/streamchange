import abc
import copy
import pandas as pd
import numpy as np
import optuna
import multiprocessing
from typing import Tuple, Callable
from numba import njit
import plotly.graph_objects as go
import plotly.express as px

from .window_segmentor import WindowSegmentor
from .estimators import SeparableAMOCEstimator


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


class BasePenaltyTuner:
    @abc.abstractmethod
    def fit(self) -> "BasePenaltyTuner":
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "penalty_scale_"):
            msg = f"This instance of {type(self).__name__} is not fitted yet."
            raise RuntimeError(msg)

    @abc.abstractmethod
    def _summarise(self) -> dict:
        return {"cpt_count": None, "penalty": None, "penalty_scale": None}

    def summarise(self) -> pd.DataFrame:
        self._check_is_fitted()
        results = self._summarise()
        return pd.DataFrame(results).sort_values("penalty_scale").reset_index(drop=True)

    def show(self, title="", xvar="penalty_scale") -> go:
        self._check_is_fitted()
        results = self.summarise()
        fig = px.scatter(results, x=xvar, y="cpt_count", title=title)
        if xvar == "penalty_scale":
            vline_value = self.detector.estimator.penalty.scale
        elif xvar == "penalty":
            vline_value = self.detector.estimator.penalty()
        fig.add_vline(vline_value, line_width=0.5, line_dash="dot")
        fig.add_hline(self.target_cpts, line_width=0.5, line_dash="dot")
        return fig


class AMOCPenaltyTuner(BasePenaltyTuner):
    """
    Class for tuning the penalty of an AMOCEstimator in WindowSegmentor.

    Parameters
    ----------
    detector :
        WindowSegmentor to tune.

    target_cpts:
        Target number of changepoints to be detected in the training data.

    """

    def __init__(
        self,
        detector: WindowSegmentor,
        target_cpts: int = 1,
        interval_generator="dyadic",
        prob: float = 0.1,
        step: int = 5,
        alpha: float = 1.5,
        step_proportion: float = 0.25,
        selector: Callable = targetscaler(alpha=1.0),
    ):
        self.detector = detector
        self.target_cpts = target_cpts
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
        penalties = np.zeros(self.target_cpts)
        i = 0
        while (i < self.target_cpts) & np.any(scores > 0.0):
            argmax = scores.argmax()
            penalties[i] = scores[argmax]
            max_cpt = cpts[argmax]
            cpt_in_interval = (max_cpt >= ends) & (max_cpt < starts)
            scores[cpt_in_interval] = 0.0
            i += 1
        return penalties

    def fit(self, x: pd.DataFrame) -> "AMOCPenaltyTuner":
        if x.shape[0] < self.target_cpts:
            raise ValueError("x must contain more rows than target_cpts.")

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
            "cpt_count": np.arange(self.target_cpts) + 1,
            "penalty": self.penalties,
            "penalty_scale": self.penalties / self.detector.estimator.penalty.value,
        }
        return results


# TODO: Generalize the Optuna tuner to any ChangeDetector or AnomalyDetector.
class _Optuna_Penalty_Objective:
    def __init__(
        self,
        target_cpts: int,
        detector: WindowSegmentor,
        x: pd.DataFrame,
        score="abs_error",
        min_penalty_scale=1e-8,
        max_penalty_scale=1e8,
    ):
        self.target_cpts = target_cpts
        self.detector = detector
        self.x = x
        self.min_penalty_scale = min_penalty_scale
        self.max_penalty_scale = max_penalty_scale

        if score == "abs_error":
            self._get_score = lambda cpt_count: abs(cpt_count - self.target_cpts)
        elif score == "squared_error":
            self._get_score = lambda cpt_count: (cpt_count - self.target_cpts) ** 2

    def __call__(self, trial: optuna.Trial):
        detector = copy.deepcopy(self.detector)
        penalty_scale = trial.suggest_float(
            "penalty_scale",
            self.min_penalty_scale,
            self.max_penalty_scale,
            log=True,
        )
        detector.estimator.penalty.scale = penalty_scale
        changepoints = detector.fit_predict(self.x)
        cpt_count = len(changepoints)
        trial.set_user_attr("cpt_count", cpt_count)
        return self._get_score(cpt_count)


class OptunaPenaltyTuner(BasePenaltyTuner):
    def __init__(
        self,
        detector: WindowSegmentor,
        target_cpts: int,
        penalty_scales: list = None,
        score="abs_error",
        interpolate=True,
        n_jobs: int = 1,
    ):
        self.detector = detector
        self.target_cpts = target_cpts
        self.penalty_scales = penalty_scales
        self.score = score
        self.interpolate = interpolate
        self.n_jobs = n_jobs

    def fit(self, x: pd.DataFrame) -> "OptunaPenaltyTuner":
        if x.shape[0] < self.target_cpts:
            raise ValueError("x must contain more rows than max_cpts.")
        if self.penalty_scales is None:
            data_scale = x.std().mean()
            self.penalty_scales = data_scale * np.geomspace(1e-3, 1e3, 100)

        self.objective = _Optuna_Penalty_Objective(
            self.target_cpts,
            self.detector,
            x,
            self.score,
        )
        search_space = {"penalty_scale": self.penalty_scales}
        sampler = optuna.samplers.GridSampler(search_space)
        # Store study for introspection
        self.study = optuna.study.create_study(sampler=sampler)
        cpu_count = multiprocessing.cpu_count()
        n_jobs = self.n_jobs if self.n_jobs >= 1 else cpu_count + self.n_jobs + 1
        self.study.optimize(self.objective, n_jobs=n_jobs)

        if not self.interpolate:
            penalty_scale_ = self.study.best_params["penalty_scale"]
        else:
            interpolated_results = self._interpolate_summary()
            penalty_scale_ = interpolated_results.loc[self.target_cpts, "penalty_scale"]
        self.penalty_scale_ = penalty_scale_
        self.detector.estimator.penalty.scale = penalty_scale_
        return self

    def _summarise(self) -> pd.DataFrame:
        trials = self.study.trials
        penalty_scales = [trial.params["penalty_scale"] for trial in trials]
        cpt_count = [trial.user_attrs["cpt_count"] for trial in trials]
        scores = [trial.values[0] for trial in trials]
        default_penalty = self.detector.estimator.penalty.default_penalty()
        results = {
            "penalty": [scale * default_penalty for scale in penalty_scales],
            "penalty_scale": penalty_scales,
            "cpt_count": cpt_count,
            self.score: scores,
        }
        return results

    def _interpolate_summary(self) -> pd.DataFrame:
        results = pd.DataFrame(self._summarise())
        results = results.sort_values("penalty_scale").reset_index(drop=True)
        unique_results = (
            results.groupby("cpt_count")
            .apply(lambda x: x.iloc[x.penalty_scale.argmin()])
            .drop("cpt_count", axis=1)
        )
        min_cpts = results.cpt_count.min()
        max_cpts = results.cpt_count.max()
        columns = unique_results.columns
        index = range(min_cpts, max_cpts + 1)
        interpolated_results = pd.DataFrame(columns=columns, index=index)
        for column in unique_results.columns:
            interpolated_results[column] = unique_results[column]
        interpolated_results.interpolate(inplace=True)
        return interpolated_results
