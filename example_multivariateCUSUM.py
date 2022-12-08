from streamchange.amoc_test import MultivariateCUSUM
from streamchange.detector import WindowSegmentor
from streamchange.utils.example_data import three_segments_data
from river.stream import iter_pandas

seg_len = 100
df = three_segments_data(p=2, seg_len=seg_len, mean_change=10)

test = MultivariateCUSUM().set_default_threshold(df.shape[0], 10 * df.shape[1])
detector = WindowSegmentor(test, min_window=4, max_window=100)
cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)

from streamchange.plot import MultivariateTimeSeriesFigure

fig = MultivariateTimeSeriesFigure(df.columns)
