from . import ChangeDetector

import pandas as pd
import numpy as np
import copy


class ChangeDetectorCollection(ChangeDetector):
    def __init__(self, names: list, base_detector: ChangeDetector):
        self.size = len(names)
        self.detectors = {name: copy.deepcopy(base_detector) for name in names}

    def __getitem__(self, name):
        return self.detector[name]

    def __iter__(self):
        for name in self.detectors:
            yield name, self[name]

    def items(self):
        return self.detectors.items()

    def values(self):
        return self.detectors.values()

    def keys(self):
        return self.detectors.keys()

    def _reset(self):
        self._change_detected = False
        self._changepoints = []
        for name in self.keys():
            self[name].reset()

    def update(self, values: dict):
        self._changepoints = []
        for name, value in values.items():
            if pd.isnull(value):
                continue
            self[name].update(value)
            self._changepoints += self[name]._changepoints

        self._change_detected = True if len(self._changepoints) > 0 else False
        return self
