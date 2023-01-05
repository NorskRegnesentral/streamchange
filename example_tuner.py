from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor
from streamchange.tune import ThresholdTuner, base_selector
from streamchange.utils.example_data import three_segments_data

seg_len = 100
df = three_segments_data(p=1, seg_len=seg_len, mean_change=2)[0]

test = UnivariateCUSUM()
detector = WindowSegmentor(test, min_window=4, max_window=100)
tune = ThresholdTuner(max_cpts=100, prob=0.1, selector=base_selector(0.5))
tune(detector, df)
tune.show()

cpts = []
for t, x in df.items():
    detector.update({df.name: x})
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)
