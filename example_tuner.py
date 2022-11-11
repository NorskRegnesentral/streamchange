from streamchange.detector import UnivariateCUSUM
from streamchange.threshold_tuner import SimpleTuner
from streamchange.utils.example_data import three_segments_data

seg_len = 10000
df = three_segments_data(p=1, seg_len=seg_len, mean_change=2)[0]

detector = UnivariateCUSUM(min_window=4, max_window=1000)
tune = SimpleTuner(0.5, max_cpts=100, sampling_probability=0.1)
tune(detector, df)
tune.show()

cpts = []
for index, value in df.items():
    detector.update(value)
    if detector._change_detected:
        cpts.append((index, detector.cpts))
print(cpts)
