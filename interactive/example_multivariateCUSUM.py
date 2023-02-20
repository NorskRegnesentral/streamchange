from river.stream import iter_pandas

from streamchange.amoc import SumCUSUM, MaxCUSUM, WindowSegmentor
from streamchange.data import simulate
from streamchange.utils.profiler import Profiler

df = simulate([0, 10, 0], [10000], p=2)
test = MaxCUSUM().set_default_threshold(df.shape[0], 10 * df.shape[1])
detector = WindowSegmentor(test, 4, 100, with_jumpback=True)

cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)

from streamchange.plot import MultivariateTimeSeriesFigure

fig = MultivariateTimeSeriesFigure(df.columns)
fig.add_raw_data(df)
fig.show()
