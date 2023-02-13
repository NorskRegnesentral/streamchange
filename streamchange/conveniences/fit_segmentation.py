import pandas as pd
import numpy as np
from collections import deque

from streamchange.detector import ChangeDetector
from streamchange.segment_stats import StatUnion


def fit_segmentation(detector: ChangeDetector, stat: StatUnion, series: pd.Series):
    segmentation = []
    for i, x in enumerate(series.values):
        detector.update(x)
        stat.update(x)
        if detector.change_detected:
            history = series.iloc[max(0, i - stat.max_history + 1) : i + 1]
            cpts = detector.changepoints
            segment_stat = stat.get(cpts[0])
            segment_stat["end"] = history.index[cpts[0]]
            segmentation.append(segment_stat)
            if len(cpts) > 1:
                for cpt, next_cpt in zip(cpts[:-1], cpts[1:]):
                    segment = history.values[cpt + 1 : next_cpt + 1]
                    segment_stat = stat.reset().update_many(segment).get()
                    segment_stat["end"] = history.index[next_cpt]
                    segmentation.append(segment_stat)
            stat.reset().update_many(history.values[cpts[-1] + 1 :])

    last_stat = stat.get()
    last_stat["end"] = series.index[-1]
    segmentation.append(last_stat)
    return segmentation
