from river.stream import iter_pandas

from streamchange.amoc import CUSUM, WindowSegmentor
from streamchange.penalties import BIC
from streamchange.data import simulate

df = simulate([0, 10, 0], [100000], p=1)
estimator = CUSUM(BIC(scale=2))
detector = WindowSegmentor(estimator, 4, 100, candidate_step=1, with_jumpback=True)
cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)
