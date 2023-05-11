from river.stream import iter_pandas

from streamchange.amoc import SumCUSUM, MaxCUSUM, WindowSegmentor
from streamchange.penalties import BIC
from streamchange.data import simulate
from streamchange.utils import Profiler

p = 2
df = simulate([0, 10, 0], [1000], p=p)
estimator = SumCUSUM(BIC(p=p))
detector = WindowSegmentor(estimator, 4, 100, minsl=4, with_jumpback=True)

cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)

# from streamchange.plot import MultivariateTimeSeriesFigure

# fig = MultivariateTimeSeriesFigure(df.columns)
# fig.add_raw_data(df)
# fig.show()
