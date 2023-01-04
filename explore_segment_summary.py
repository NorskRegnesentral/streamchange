from river.stats import Mean, Quantile
import pandas as pd
import numpy as np

from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor
from streamchange.stats import StatCollection, Buffer
from streamchange.utils.example_data import three_segments_data

seg_len = 100000
df = three_segments_data(p=1, seg_len=seg_len, mean_change=10)[0]

##################
## Single stat ##
##################
test = UnivariateCUSUM().set_default_threshold(10 * df.size)
max_window = 100
detector = WindowSegmentor(test, min_window=4, max_window=max_window)
stat = Buffer(Mean(), max_window)

cpts = []
segment_stats = []
for t, x in df.items():
    detector.update({df.name: x})
    stat.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
        detector_window = detector.get_window(as_pandas=False)
        new_stats = stat.get_restart(detector.changepoints, detector_window)
        segment_stats += new_stats
segment_stats.append(stat.get())
print(cpts)
print(segment_stats)


###################
## Several stats ##
###################
test = UnivariateCUSUM().set_default_threshold(10 * df.size)
max_window = 100
detector = WindowSegmentor(test, min_window=4, max_window=max_window)
stat = StatCollection({
    "mean": Buffer(Mean(), max_window),
    "quantile01": Buffer(Quantile(0.01), max_window),
    "quantile99": Buffer(Quantile(0.99), max_window),
})

cpts = []
segment_stats = []
for t, x in df.items():
    detector.update({df.name: x})
    stat.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
        detector_window = detector.get_window(as_pandas=False)
        new_stats = stat.get_restart(detector.changepoints, detector_window)
        segment_stats += new_stats
segment_stats.append(stat.get())
print(cpts)
print(segment_stats)
print(pd.DataFrame(segment_stats))



# Custom needs/utilities:
#  - stat.revert() for all stats.
#  - stat.restart(x_aftercpt): reset stat, then .update_many().
#  - Implement these functions on collection level to avoid the visible loops over stats?
#      * Similar to Transformer Collection?
#  - Pipeline: ChangeDetector -> stats?
#     * Single .update, .
#     * Similar to Transformer Pipeline?
