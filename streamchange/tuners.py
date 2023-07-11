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
    def fit(self) -> "BasePenaltyTuner":
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
            vline_value = self.detector.get_penalty().scale
        elif xvar == "penalty":
            vline_value = self.detector.get_penalty()()
        fig.add_vline(vline_value, line_width=0.5, line_dash="dot")
        fig.add_hline(self.target_cpts, line_width=0.5, line_dash="dot")
        return fig


class _Optuna_Penalty_Objective:
    def __init__(
        self,
        target_cpts: int,
        detector: ChangeDetector,
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
        detector.get_penalty().scale = penalty_scale
        changepoints = detector.fit_predict(self.x)
        cpt_count = len(changepoints)
        trial.set_user_attr("cpt_count", cpt_count)
        return self._get_score(cpt_count)


class GridPenaltyTuner(BasePenaltyTuner):
    def __init__(
        self,
        detector: ChangeDetector,
        target_cpts: int,
        penalty_scales: list = None,
        score="abs_error",
        interpolate=True,
        n_jobs: int = 1,
    ):
        super().__init__(detector)
        self.target_cpts = target_cpts
        self.penalty_scales = penalty_scales
        self.score = score
        self.interpolate = interpolate
        self.n_jobs = n_jobs

    def _summarise(self) -> pd.DataFrame:
        trials = self.study.trials
        penalty_scales = [trial.params["penalty_scale"] for trial in trials]
        cpt_count = [trial.user_attrs["cpt_count"] for trial in trials]
        scores = [trial.values[0] for trial in trials]
        default_penalty = self.detector.get_penalty().default_penalty()
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
        index = np.arange(min_cpts, max_cpts + 1)
        interpolated_results = pd.DataFrame(columns=columns, index=index)
        for column in unique_results.columns:
            interpolated_results[column] = unique_results[column]
        interpolated_results.interpolate(inplace=True)
        return interpolated_results

    def fit(self, x: pd.DataFrame) -> "GridPenaltyTuner":
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
            results = self._interpolate_summary()
            best_index = results.abs_error.idxmin()
            penalty_scale_ = results.loc[best_index, "penalty_scale"]
        self.penalty_scale_ = penalty_scale_
        self.detector_ = copy.deepcopy(self.detector)
        self.detector_.get_penalty().scale = penalty_scale_
        self.detector_.fit(x)
        return self
