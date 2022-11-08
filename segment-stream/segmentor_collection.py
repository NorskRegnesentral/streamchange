from .segmentor import Segmentor

import pandas as pd
import numpy as np
import copy


class SegmentorCollection:
    def __init__(self, names: list, base_segmentor: Segmentor):
        self.size = len(names)
        self.segmentors = {name: copy.deepcopy(base_segmentor) for name in names}

    def __getitem__(self, name):
        return self.segmentors[name]

    def __iter__(self):
        for name in self.segmentors:
            yield name, self[name]

    def items(self):
        return self.segmentors.items()

    def values(self):
        return self.segmentors.values()

    def keys(self):
        return self.segmentors.keys()

    def update_fit_dict_int(self, values: dict, timestamp: int):
        for name, value in values.items():
            if pd.isnull(value):
                continue
            self[name].update_fit_float_int(value, timestamp)
        return self

    def update_fit_pddf(self, df: pd.DataFrame):
        for column, series in df.items():
            series = series.dropna()
            if series.size == 0:
                continue
            self[column].update_fit_pdseries(series)
        return self

    def update_fit(self, values, timestamps=None, name: str = None):
        if not name is None:
            self[name].update_fit(values, timestamps)
        elif isinstance(timestamps, int):
            self.update_fit_dict_int(values, timestamps)
        elif isinstance(values, pd.DataFrame):
            self.update_fit_pddf(values)
        else:
            raise ValueError(
                "values must be either a pandas.DataFrame, a dictionary-like object"
                " representing one row of observations (a pd.Series for example) where"
                " the key represents the variable or segmentor name, or."
            )
        return self

    def model_at(self, times: list) -> pd.DataFrame:
        models = []
        for name, segmentor in self:
            current_model = segmentor.model_at(times)
            current_model.insert(0, "variable", name)
            models.append(current_model)
        return pd.concat(models)

    def wide_model_at(self, times: list) -> pd.DataFrame:
        model = self.model_at(times)
        model["time"] = model.index
        wide_model = model.pivot(index="time", columns="variable")
        return wide_model

    def model_now(self, parameter: str) -> pd.DataFrame:
        models = {name: [segmentor.model_now(parameter)] for name, segmentor in self}
        last_timestamps = [segmentor.timestamps[-1] for segmentor in self.values()]
        return pd.DataFrame(models, index=[max(last_timestamps)])

    def model_now_as_numpy(self, parameter: str) -> np.ndarray:
        param_values = [segmentor.model_now(parameter) for segmentor in self.values()]
        return np.array(param_values).reshape(1, self.size)

    def changepoint_timestamps(self):
        cpt_timestamps = [
            segment["start"]
            for segmentor in self.values()
            for segment in segmentor.model
        ]
        cpt_timestamps = list(set(cpt_timestamps))
        cpt_timestamps.sort()
        return cpt_timestamps
