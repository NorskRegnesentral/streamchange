from river.stream import iter_pandas

from streamchange.amoc_test import SumCUSUM, MaxCUSUM
from streamchange.detector import WindowSegmentor, JumpbackWindow
from streamchange.utils.example_data import three_segments_data
from streamchange.utils.profiler import Profiler

seg_len = 100000
df = three_segments_data(p=10, seg_len=seg_len, mean_change=10)

test = SumCUSUM().set_default_threshold(df.shape[0], 10 * df.shape[1])
window = JumpbackWindow(4, 100)
detector = WindowSegmentor(test, window)

cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)

from streamchange.plot import MultivariateTimeSeriesFigure

fig = MultivariateTimeSeriesFigure(df.columns)
