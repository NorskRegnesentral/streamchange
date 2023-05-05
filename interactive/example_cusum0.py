from river.stream import iter_pandas

from streamchange.amoc import CUSUM0, WindowSegmentor
from streamchange.data import simulate

df = simulate([0, 10, 0], [100], p=1)
estimator = CUSUM0()
detector = WindowSegmentor(estimator, 4, 500, minsl=1, with_jumpback=True)
cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)
