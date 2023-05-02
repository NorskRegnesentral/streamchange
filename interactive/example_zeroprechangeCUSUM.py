from river.stream import iter_pandas

from streamchange.amoc import ZeroPrechangeCUSUM, WindowSegmentor
from streamchange.data import simulate

df = simulate([0, 10, 0], [100], p=1)
test = ZeroPrechangeCUSUM(minsl=1).set_default_threshold(df.size)
detector = WindowSegmentor(test, 4, 100, with_jumpback=False)
cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
# print(cpts)
