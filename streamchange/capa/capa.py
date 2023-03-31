import numpy as np
import pandas as pd
from numbers import Number
from typing import Tuple

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
    ):
        assert minsl >= 2
        assert maxsl > minsl
        self.minsl = minsl
        self.maxsl = maxsl
        self.csaving = csaving
        self.psaving = psaving if not psaving is None else csaving
        self.reset()

    def reset(self) -> "Capa":
        self.window = NumpyDeque(self.maxsl)
        self.opt_saving = NumpyDeque(self.maxsl)
        self.opt_saving.append(0)
        self.anom_start = 0
        return self

    def update(self, x: Number) -> "Capa":
        self.window.append(x)
        base_saving = self.opt_saving.values[-1]
        collective_saving = 0
        point_saving = 0
        n = len(self.window)
        if n >= self.minsl:
            opt_savings = self.opt_saving.values[: -self.minsl + 1]
            csavings = self.csaving.cumopt(self.window.values)[: -self.minsl + 1]
            candidate_savings = opt_savings + csavings
            cpt = np.argmax(candidate_savings) - n - 1
            collective_saving = candidate_savings[cpt + n + 1]
            point_saving = base_saving + self.psaving.opt(x)

        savings = [base_saving, point_saving, collective_saving]
        argmax = np.argmax(savings)
        self.opt_saving.append(savings[argmax])
        self.anom_start = cpt + 1 if argmax == 2 else -argmax
        return self

    def fit(self, x: pd.Series) -> "Capa":
        self.reset()
        x = x.dropna()
        anom_starts = []
        for value in x.values:
            self.update(value)
            anom_starts.append(self.anom_start)
        if len(anom_starts) == 0:
            # To silence FutureWarning about dtypes for empty series.
            self.anom_starts = pd.Series(anom_starts, dtype=x.index.dtype)
        else:
            self.anom_starts = pd.Series(anom_starts, index=x.index)
        return self

    def predict(self) -> Tuple[list, list]:
        collective_anoms = self.collective_anomalies(self.anom_starts)
        point_anoms = self.point_anomalies(self.anom_starts)
        return collective_anoms, point_anoms

    @property
    def point_anomaly_detected(self) -> bool:
        return self.anom_start == -1

    @property
    def collective_anomaly_detected(self) -> bool:
        return self.anom_start < -1

    @property
    def anomaly_detected(self) -> bool:
        return self.point_anomaly_detected() or self.collective_anomaly_detected()

    @staticmethod
    def collective_anomalies(anom_starts: pd.Series) -> list:
        i = -1
        times = anom_starts.index
        starts = anom_starts.values
        anoms = []
        while i >= -starts.size:
            start_i = starts[i]
            if start_i < -1:
                anoms.append(
                    {
                        "start": times[i + 1 + start_i],
                        "end": times[i],
                        "size": abs(start_i),
                    }
                )
                i += start_i + 1
            i -= 1
        return anoms

    @staticmethod
    def point_anomalies(anom_starts: pd.Series) -> list:
        i = -1
        times = anom_starts.index
        starts = anom_starts.values
        anoms = []
        while i >= -starts.size:
            start_i = starts[i]
            if start_i < -1:
                i += start_i + 1
            elif start_i == -1:
                time_i = times[i]
                anoms.append({"start": time_i, "end": time_i, "size": 1})
            i -= 1
        return anoms
