import numpy as np
import pandas as pd
from numbers import Number
from typing import Tuple
import copy

from streamchange.base import NumpyDeque
from .savings import BaseSaving, ConstMeanL2


class Capa:
    # Assumes x is standardised outside of this class.
    def __init__(
        self,
        csaving: BaseSaving = ConstMeanL2(),
        psaving: BaseSaving = None,
        minsl=2,
        maxsl=1000,
        predict_point_anomalies=False,
    ):
        assert minsl >= 2
        assert maxsl > minsl
        self.minsl = minsl
        self.maxsl = maxsl
        self.csaving = csaving
        self.psaving = psaving if not psaving is None else csaving
        self.predict_point_anomalies = predict_point_anomalies
        self.reset()

    def reset(self) -> "Capa":
        self.window = NumpyDeque(self.maxsl)
        self.opt_saving = NumpyDeque(self.maxsl)
        self.opt_saving.appendleft(0)
        self.anomaly_start = 0
        return self

    def get_penalty(self):
        return self.csaving.penalty

    @property
    def point_anomaly_detected(self) -> bool:
        return self.anomaly_start == 0

    @property
    def collective_anomaly_detected(self) -> bool:
        return self.anomaly_start > 0

    @property
    def anomaly_detected(self) -> bool:
        return self.point_anomaly_detected() or self.collective_anomaly_detected()

    def update(self, x: Number) -> "Capa":
        self.window.appendleft(x)
        base_saving = self.opt_saving.values[0]
        collective_saving = 0
        point_saving = 0
        n = len(self.window)
        if n >= self.minsl:
            opt_savings = self.opt_saving.values[self.minsl - 1 :]
            csavings = self.csaving.cumopt(self.window.values)[self.minsl - 1 :]
            candidate_savings = opt_savings + csavings
            argmax_candidate = np.argmax(candidate_savings)
            cpt = self.minsl + argmax_candidate
            collective_saving = candidate_savings[argmax_candidate]
            point_saving = base_saving + self.psaving.opt(x)

        savings = [base_saving, point_saving, collective_saving]
        argmax = np.argmax(savings)
        self.opt_saving.appendleft(savings[argmax])
        if argmax == 2:
            self.anomaly_start = cpt - 1
        elif argmax == 1:
            self.anomaly_start = 0
        else:
            self.anomaly_start = -1
        return self

    def fit(self, x: pd.Series) -> "Capa":
        self.reset()
        x = x.dropna()
        anomaly_starts = []
        for value in x.values:
            self.update(value)
            anomaly_starts.append(self.anomaly_start)
        anomaly_starts = pd.Series(anomaly_starts, index=x.index, dtype=int)
        self.collective_anomalies_ = self.extract_collective_anomalies(anomaly_starts)
        self.point_anomalies_ = self.extract_point_anomalies(anomaly_starts)
        return self

    def _check_is_fitted(self):
        if not hasattr(self, "collective_anomalies_"):
            msg = f"This instance of {type(self).__name__} is not fitted yet."
            raise RuntimeError(msg)

    def predict(self, x: pd.Series = None) -> list:
        self._check_is_fitted()
        if x is None:
            anomalies = copy.deepcopy(self.collective_anomalies_)
            if self.predict_point_anomalies:
                anomalies += self.point_anomalies_
            return anomalies
        else:
            # TODO: Complete
            raise RuntimeError(
                "Prediction for new observations is not implemented yet."
            )

    def fit_predict(self, x: pd.DataFrame) -> list:
        return self.fit(x).predict()

    @staticmethod
    def extract_collective_anomalies(anomaly_starts: pd.Series) -> list:
        i = -1
        times = anomaly_starts.index
        starts = anomaly_starts.values
        anoms = []
        while i >= -starts.size:
            start_i = starts[i]
            if start_i > 0:
                anoms.append(
                    {
                        "start": times[i - start_i],
                        "end": times[i],
                        "size": abs(start_i + 1),
                    }
                )
                i -= start_i
            i -= 1
        return anoms

    @staticmethod
    def extract_point_anomalies(anomaly_starts: pd.Series) -> list:
        i = -1
        times = anomaly_starts.index
        starts = anomaly_starts.values
        anoms = []
        while i >= -starts.size:
            start_i = starts[i]
            if start_i > 0:
                i -= start_i
            elif start_i == 0:
                time_i = times[i]
                anoms.append({"start": time_i, "end": time_i, "size": 1})
            i -= 1
        return anoms
