import abc
import copy
import multiprocessing
import optuna
import pandas as pd
from numbers import Number
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Union

from .base import ChangeDetector


class BasePenaltyTuner:
    def __init__(self, detector: ChangeDetector):
        self.detector = detector

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame) -> "BasePenaltyTuner":
        self.detector_ = copy.deepcopy(self.detector)
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "detector_"):
            msg = f"This instance of {type(self).__name__} is not fitted yet."
            raise RuntimeError(msg)

    def update(self, x: Union[Number, dict]) -> "ChangeDetector":
        self.detector_.update(x)
        return self

    def predict(self, x: pd.DataFrame = None) -> np.ndarray:
        self._check_is_fitted()
        return self.detector_.predict(x)

    @abc.abstractmethod
    def _summarise(self) -> dict:
        return {"detection_count": None, "penalty": None, "penalty_scale": None}

    def summarise(self) -> pd.DataFrame:
        self._check_is_fitted()
        summary = self._summarise()
        return pd.DataFrame(summary).sort_values("penalty_scale").reset_index(drop=True)

    def show(self, title="", xvar="penalty_scale") -> go:
        self._check_is_fitted()
        summary = self.summarise()
        fig = px.scatter(summary, x=xvar, y="detection_count", title=title)
        if xvar == "penalty_scale":
            vline_value = self.detector.get_penalty().scale
        elif xvar == "penalty":
            vline_value = self.detector.get_penalty()()
        fig.add_vline(vline_value, line_width=0.5, line_dash="dot")
        fig.add_hline(self.target_detections, line_width=0.5, line_dash="dot")
        return fig


class _Optuna_Penalty_Objective:
    def __init__(
        self,
        target_detections: int,
        detector: ChangeDetector,
        x: pd.DataFrame,
        score="abs_error",
        min_penalty_scale=1e-8,
        max_penalty_scale=1e8,
    ):
        self.target_detections = target_detections
        self.detector = detector
        self.x = x
        self.min_penalty_scale = min_penalty_scale
        self.max_penalty_scale = max_penalty_scale

        if score == "abs_error":
            self._get_score = lambda detection_count: abs(
                detection_count - self.target_detections
            )
        elif score == "squared_error":
            self._get_score = (
                lambda detection_count: (detection_count - self.target_detections) ** 2
            )

    def __call__(self, trial: optuna.Trial):
        detector = copy.deepcopy(self.detector)
        penalty_scale = trial.suggest_float(
            "penalty_scale",
            self.min_penalty_scale,
            self.max_penalty_scale,
            log=True,
        )
        detector.get_penalty().scale = penalty_scale
        detections = detector.fit_predict(self.x)
        detection_count = len(detections)
        trial.set_user_attr("detection_count", detection_count)
        return self._get_score(detection_count)


class GridPenaltyTuner(BasePenaltyTuner):
    def __init__(
        self,
        detector: ChangeDetector,
        target_detections: int,
        penalty_scales: list = None,
        score="abs_error",
        interpolate=True,
        n_jobs: int = 1,
        refit=True,
    ):
        super().__init__(detector)
        self.target_detections = target_detections
        self.penalty_scales = penalty_scales
        self.score = score
        self.interpolate = interpolate
        self.n_jobs = n_jobs
        self.refit = refit

    def _summarise(self) -> pd.DataFrame:
        trials = self.study.trials
        penalty_scales = [trial.params["penalty_scale"] for trial in trials]
        detection_count = [trial.user_attrs["detection_count"] for trial in trials]
        scores = [trial.values[0] for trial in trials]
        default_penalty = self.detector.get_penalty().default_penalty()
        summary = {
            "penalty": [scale * default_penalty for scale in penalty_scales],
            "penalty_scale": penalty_scales,
            "detection_count": detection_count,
            self.score: scores,
        }
        return summary

    def _interpolate_summary(self) -> pd.DataFrame:
        summary = pd.DataFrame(self._summarise())
        summary = summary.sort_values("penalty_scale").reset_index(drop=True)
        unique_summary = (
            summary.groupby("detection_count")
            .apply(lambda x: x.iloc[x.penalty_scale.argmin()])
            .drop("detection_count", axis=1)
        )
        min_detections = summary.detection_count.min()
        max_detections = summary.detection_count.max()
        columns = unique_summary.columns
        index = np.arange(min_detections, max_detections + 1)
        interpolated_summary = pd.DataFrame(columns=columns, index=index)
        for column in unique_summary.columns:
            interpolated_summary[column] = unique_summary[column]
        interpolated_summary.interpolate(inplace=True)
        return interpolated_summary

    def fit(self, x: pd.DataFrame) -> "GridPenaltyTuner":
        if x.shape[0] < self.target_detections:
            raise ValueError("x must contain more rows than max_detections.")
        if self.penalty_scales is None:
            data_scale = x.std().mean()
            self.penalty_scales = data_scale * np.geomspace(1e-3, 1e3, 100)

        self.objective = _Optuna_Penalty_Objective(
            self.target_detections,
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
            self.summary = self._summarise()
            penalty_scale_ = self.study.best_params["penalty_scale"]
        else:
            self.interpolated_summary = self._interpolate_summary()
            best_index = self.interpolated_summary.abs_error.idxmin()
            penalty_scale_ = self.interpolated_summary.loc[best_index, "penalty_scale"]
        self.penalty_scale_ = penalty_scale_

        self.detector_ = copy.deepcopy(self.detector)
        self.detector_.get_penalty().scale = penalty_scale_
        if self.refit:
            self.detector_.fit(x)

        return self
