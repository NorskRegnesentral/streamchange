from river.stream import iter_pandas

from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor
from streamchange.tune import ThresholdTuner, base_selector
from streamchange.data import simulate

df = simulate([0, 10, 0], [100], p=1)
test = UnivariateCUSUM(minsl=1).set_default_threshold(10 * df.size)
detector = WindowSegmentor(test, 4, 100)
tune = ThresholdTuner(max_cpts=100, prob=0.1, selector=base_selector(0.5))
tune(detector, df)
tune.show()

cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)
