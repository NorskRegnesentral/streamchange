from river.stats import Mean, Quantile
import pandas as pd
import numpy as np

from streamchange.amoc import WindowSegmentor, CUSUM
from streamchange.segment_stats import StatUnion, StatBuffer
from streamchange.conveniences import fit_segmentation
from streamchange.data import simulate
from streamchange.penalties import BIC

seg_len = 50
series = simulate([0, 10, 0], [seg_len], p=1)[0]


#################
## Single stat ##
#################
estimator = CUSUM(penalty=BIC(scale=10))
detector = WindowSegmentor(estimator, 2, 100)
stat = StatUnion({"mean": StatBuffer(Mean())}, detector.max_window)
segmentation = fit_segmentation(detector, stat, series)
print(pd.DataFrame(segmentation))


###################
## Several stats ##
###################
estimator = CUSUM(penalty=BIC(scale=10))
detector = WindowSegmentor(estimator, 4, 100)
stat = StatUnion(
    {
        "mean": StatBuffer(Mean()),
        "quantile01": StatBuffer(Quantile(0.01)),
        "quantile99": StatBuffer(Quantile(0.99)),
    },
    detector.max_window,
)
segmentation = fit_segmentation(detector, stat, series)
print(pd.DataFrame(segmentation))


# Custom needs/utilities:
#  - Implement these functions on collection level to avoid the visible loops over stats?
#      * Similar to Transformer Collection?
#  - Pipeline: ChangeDetector -> stats?
#     * Single .update, .
#     * Similar to Transformer Pipeline?
