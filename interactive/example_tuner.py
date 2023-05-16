from river.stream import iter_pandas

from streamchange.amoc import (
    CUSUM,
    WindowSegmentor,
    PenaltyTuner,
    base_selector,
)
from streamchange.data import simulate

df = simulate([0, 10, 0], [10000], p=1)
detector = WindowSegmentor(CUSUM(), 4, 100)
tuner = PenaltyTuner(
    detector, max_cpts=100, prob=0.1, max_window_only=False, selector=base_selector(0.5)
)
tuner.fit(df)
tuner.show()

cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)


from streamchange.amoc import OptunaPenaltyTuner
import numpy as np

df = simulate([0, 10, 0], [1000], p=1)
detector = WindowSegmentor(CUSUM(), 4, 100)
penalty_scales = np.geomspace(1e-2, 10, 20)
tuner = OptunaPenaltyTuner(detector, 100, penalty_scales)
tuner.fit(df)
tuner.show()
