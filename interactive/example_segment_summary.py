from river.stats import Mean, Quantile
import pandas as pd
import numpy as np

from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor
from streamchange.segment_stats import StatCollection, Buffer
from streamchange.conveniences import fit_segmentation
from streamchange.data import simulate

seg_len = 10000
series = simulate([0, 10, 0], [seg_len], p=1)[0]


#################
## Single stat ##
#################
test = UnivariateCUSUM().set_default_threshold(2)
detector = WindowSegmentor(test, 4, 100)
stat = StatCollection({"mean": Buffer(Mean(), detector.max_window)})
segmentation = fit_segmentation(detector, stat, series)
print(pd.DataFrame(segmentation))


###################
## Several stats ##
###################
test = UnivariateCUSUM().set_default_threshold(10 * series.size)
detector = WindowSegmentor(test, 4, 100)
stat = StatCollection(
    {
        "mean": Buffer(Mean(), detector.max_window),
        "quantile01": Buffer(Quantile(0.01), detector.max_window),
        "quantile99": Buffer(Quantile(0.99), detector.max_window),
    }
)
segmentation = fit_segmentation(detector, stat, series)
print(pd.DataFrame(segmentation))


# Custom needs/utilities:
#  - Implement these functions on collection level to avoid the visible loops over stats?
#      * Similar to Transformer Collection?
#  - Pipeline: ChangeDetector -> stats?
#     * Single .update, .
#     * Similar to Transformer Pipeline?
