from river.stats import Mean, Quantile
from river.stream import iter_pandas
import pandas as pd
import numpy as np

from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor, JumpbackWindow
from streamchange.segment_stats import SegmentStat, StatCollection, Buffer
from streamchange.data import simulate

seg_len = 10000
series = simulate([0, 10, 0], [100000], p=1)[0]


def fit_segmentation(detector: WindowSegmentor, stat: SegmentStat, series: pd.Series):
    cpts = []
    segment_stats = []
    for t, x in series.items():
        detector.update(x)
        stat.update(x)
        if detector.change_detected:
            cpts.append((t, detector.changepoints))
            new_stats = stat.get_restart(detector.changepoints, detector.window.get())
            segment_stats += new_stats
    segment_stats.append(stat.get())
    return cpts, segment_stats


#################
## Single stat ##
#################
test = UnivariateCUSUM().set_default_threshold(10 * series.size)
window = JumpbackWindow(4, 100)
detector = WindowSegmentor(test, window)
stat = Buffer(Mean(), window.max_length)
cpts, segment_stats = fit_segmentation(detector, stat, series)
print(cpts)
print(segment_stats)


###################
## Several stats ##
###################
test = UnivariateCUSUM().set_default_threshold(10 * series.size)
window = JumpbackWindow(4, 100)
detector = WindowSegmentor(test, window)
stat = StatCollection(
    {
        "mean": Buffer(Mean(), window.max_length),
        "quantile01": Buffer(Quantile(0.01), window.max_length),
        "quantile99": Buffer(Quantile(0.99), window.max_length),
    }
)
cpts, segment_stats = fit_segmentation(detector, stat, series)
print(cpts)
print(segment_stats)
print(pd.DataFrame(segment_stats))


# Custom needs/utilities:
#  - Implement these functions on collection level to avoid the visible loops over stats?
#      * Similar to Transformer Collection?
#  - Pipeline: ChangeDetector -> stats?
#     * Single .update, .
#     * Similar to Transformer Pipeline?
