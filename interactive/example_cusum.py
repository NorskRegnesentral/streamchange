from river.stream import iter_pandas

from streamchange.amoc import CUSUM, WindowSegmentor
from streamchange.data import simulate

df = simulate([0, 10, 0], [100000], p=1)
estimator = CUSUM(arl=10 * df.shape[0])
detector = WindowSegmentor(estimator, 4, 1000, candidate_step=10, with_jumpback=True)
cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)
