from river.stream import iter_pandas

from streamchange.amoc import (
    CUSUM,
    WindowSegmentor,
    ThresholdTuner,
    base_selector,
)
from streamchange.data import simulate

df = simulate([0, 10, 0], [1000], p=1)
detector = WindowSegmentor(CUSUM(), 4, 100)
tune = ThresholdTuner(
    max_cpts=100, prob=0.1, max_window_only=False, selector=base_selector(0.5)
)
tune(detector, df)
tune.show()

cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)
