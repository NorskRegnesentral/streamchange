from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor
from streamchange.utils.example_data import three_segments_data

seg_len = 100000
df = three_segments_data(p=1, seg_len=seg_len, mean_change=10)[0]

test = UnivariateCUSUM().set_default_threshold(10 * df.size)
detector = WindowSegmentor(test, min_window=4, max_window=100, fetch_test_results=True)
cpts = []
for t, x in df.items():
    detector.update({df.name: x})
    if detector.change_detected:
        cpts.append((t, detector.changepoints, detector.test_results))
print(cpts)
