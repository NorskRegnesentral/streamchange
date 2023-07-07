from river.stream import iter_pandas

from streamchange.amoc import CUSUM, WindowSegmentor, AMOCPenaltyTuner

df = simulate([0, 10, 0], [10000], p=1)
detector = WindowSegmentor(CUSUM(), 4, 100)
tuner = AMOCPenaltyTuner(detector, target_cpts=100, interval_generator="dyadic")
tuner.fit(df)
tuner.show()

cpts = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
print(cpts)


from streamchange.data import simulate
from streamchange.amoc import CUSUM, WindowSegmentor, AMOCPenaltyTuner
from streamchange.tuners import GridPenaltyTuner
import numpy as np

df = simulate([0, 10, 0], [1000], p=1)
detector = WindowSegmentor(CUSUM(), 4, 100)
penalty_scales = np.geomspace(1e-2, 10, 20)
detector = GridPenaltyTuner(detector, 100, penalty_scales)
detector.fit(df)
detector.show()

cpts = detector.predict(df)
print(cpts)
