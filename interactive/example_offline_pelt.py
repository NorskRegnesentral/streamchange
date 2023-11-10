import numpy as np
import pandas as pd
from streamchange.data import simulate
from streamchange.pelt import Pelt, L2Cost
from streamchange.offline.pelt import OfflinePelt
from streamchange.offline.costs import OfflineL2Cost

# Online
df = simulate([0, 10, 0, 10, 0], [10])
detector = Pelt(L2Cost(), minsl=2, maxsl=10000)
segments = detector.fit_predict(df)
print(pd.DataFrame(segments))

# Offline
df = simulate([0, 10, 0, 10, 0], [10], p=2)
detector = OfflinePelt(OfflineL2Cost(), minsl=2)
detector.fit(df)
print(detector.segments_)

from streamchange.utils import Profiler

df = simulate([0, 10], [10000], p=1)
detector = OfflinePelt(OfflineL2Cost(), minsl=2)
profiler = Profiler()
profiler.start()
detector.fit(df)
profiler.stop()
print(detector.segments_)

detector = Pelt(L2Cost(), minsl=2, maxsl=1000000)
segments = detector.fit_predict(df)
print(pd.DataFrame(segments))
