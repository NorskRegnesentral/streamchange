import pandas as pd
import numpy as np
from numba import njit
import plotly.graph_objects as go
import plotly.express as px
import optuna
import multiprocessing
import copy

from .window_segmentor import WindowSegmentor


@njit
def generate_intervals(
    data_size: int,
    min_window: int,
    max_window: int,
    prob: float = 1.0,
):
    starts = []
    ends = []
    for end in range(min_window, data_size):
        min_start = max(0, end - max_window)
        max_start = end - min_window + 1
        for start in range(min_start, max_start):
            if np.random.uniform(0.0, 1.0) <= prob:
                starts.append(start)
                ends.append(end)
    return np.array(starts), np.array(ends)


def base_selector(alpha=0.0):
    def selector(thresholds):
        return max((1 + alpha) * thresholds[-1], 1e-8)

    return selector


# TODO: Update ThresholdTuner with new cpt definition and new WindowSegmentor


class PenaltyTuner:
    """
    Class for tuning WindowSegmentor detectors with SeparableAMOCEstimators.
    """

    def __init__(
        self,
        detector: WindowSegmentor,
        max_cpts: int = 1000,
        prob: float = 1.0,
        max_window_only=True,
        selector=base_selector(),
    ):
        self.detector = detector
        self.max_cpts = max_cpts
        self.prob = prob
        self.max_window_only = max_window_only
        self.selector = selector

    def fit(self, x: pd.DataFrame):
        if x.shape[0] < self.max_cpts:
            raise ValueError("x must contain more rows than max_cpts.")

        self.x = x.to_numpy()
        self._find_thresholds()
        self.detector.estimator.penalty = self.selector(self.thresholds)

    def _detect_in(self, starts: list, ends: list):
        """
        Outputs the optimal test statistic and changepoint of each interval given
        by start[i]:(end[i]) in x.
        """
        scores = []
        cpts = []
        for start, end in zip(starts, ends):
            candidate_cpts = self.detector.candidate_cpts[
                self.detector.candidate_cpts < end - start
            ]
            self.detector.estimator.fit(self.x[start:end], candidate_cpts)
            scores.append(self.detector.estimator.score)
            cpts.append(start + self.detector.estimator.changepoint - 1)
        return np.array(scores), np.array(cpts)

    def _find_thresholds(self) -> np.ndarray:
        max_window = self.detector.max_window
        min_window = max_window if self.max_window_only else self.detector.min_window
        n = self.x.shape[0]
        starts, ends = generate_intervals(n, min_window, max_window, self.prob)
        scores, cpts = self._detect_in(starts, ends)

        self.thresholds = np.zeros(self.max_cpts)
        i = 0
        while (i < self.max_cpts) & np.any(scores > 0.0):
            argmax = scores.argmax()
            self.thresholds[i] = scores[argmax]
            max_cpt = cpts[argmax]
            cpt_in_interval = (max_cpt >= starts) & (max_cpt < ends)
            scores[cpt_in_interval] = 0.0
            i += 1

    def show(self, title="") -> go:
        fig = go.Figure(
            layout=go.Layout(
                title=title,
                xaxis_title="Number of changepoints",
                yaxis_title="Threshold",
            )
        )
        fig.add_traces(
            [
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=self.thresholds,
                    mode="markers",
                    name="Threshold series",
                ),
                go.Scatter(
                    x=np.arange(self.max_cpts),
                    y=np.repeat(self.detector.estimator.penalty, self.max_cpts),
                    mode="lines",
                    line_dash="dot",
                    name="Tuned threshold",
                ),
            ]
        )
        return fig


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


class OptunaPenaltyTuner:
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

    def fit(self, x: pd.DataFrame):
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

    def summarise(self) -> pd.DataFrame:
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
        return pd.DataFrame(results).sort_values("penalty_scale").reset_index(drop=True)

    def _interpolate_summary(self) -> pd.DataFrame:
        results = self.summarise()
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

    def show(self, title="", xvar="penalty_scale") -> go:
        results = self.summarise()
        fig = px.scatter(results, x=xvar, y="cpt_count", title=title)
        if xvar == "penalty_scale":
            vline_value = self.detector.estimator.penalty.scale
        elif xvar == "penalty":
            vline_value = self.detector.estimator.penalty()
        fig.add_vline(vline_value, line_width=0.5, line_dash="dot")
        fig.add_hline(self.target_cpts, line_width=0.5, line_dash="dot")
        return fig
