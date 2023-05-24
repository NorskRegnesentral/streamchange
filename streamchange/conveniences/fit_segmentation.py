import pandas as pd
import numpy as np

from streamchange.base import ChangeDetector
from streamchange.segment_stats import StatUnion


def fit_segmentation(detector: ChangeDetector, stat: StatUnion, series: pd.Series):
    """Fit a segmentation to the data

    Both estimates changepoints and segment parameters in an online fashion.
    """
    detector.reset()
    stat.reset()

    # Initial dummy segment to get first segment start in loop without if/else.
    init_stat = stat.get()
    init_stat.update({"start": -1, "end": -1})
    segmentation = [init_stat]
    for i, x in enumerate(series.values):
        detector.update(x)
        stat.update(x)
        if detector.change_detected:
            for cpt in detector.changepoints:
                segment_stat = stat.get(cpt)
                segment_stat["start"] = segmentation[-1]["end"] + 1
                segment_stat["end"] = i - cpt
                segmentation.append(segment_stat)
                post_cpt_values = series.values[i - cpt + 1 : i + 1]
                segment_stat = stat.reset().update_many(post_cpt_values).get()

    last_stat = stat.get()
    last_stat["start"] = segmentation[-1]["end"] + 1
    last_stat["end"] = series.size - 1
    segmentation.append(last_stat)
    segmentation.pop(0)  # Initial segment is just a dummy.

    return segmentation
