import numpy as np
import pandas as pd
from streamchange.data import simulate
from streamchange.pelt import Pelt as OnlinePelt, L2Cost as OnlineL2Cost
from streamchange.offline.pelt import Pelt as OfflinePelt, L2Cost as OfflineL2Cost

# Online
df = simulate([0, 10, 0, 10, 0], [10])
detector = OnlinePelt(OnlineL2Cost(), minsl=2, maxsl=10000)
segments = detector.fit_predict(df)
print(pd.DataFrame(segments))

# Offline
df = simulate([0, 10, 0, 10, 0], [10], p=2)
detector = OfflinePelt(OfflineL2Cost(), minsl=2)
detector.fit(df)
print(detector.segments_)

from streamchange.utils import Profiler

n = int(1e6)
df = simulate([0, 10], [n], p=1)
detector = OfflinePelt(OfflineL2Cost(), minsl=2)
profiler = Profiler()
profiler.start()
detector.fit(df)
profiler.stop()
print(detector.segments_)

detector = OnlinePelt(OnlineL2Cost(), minsl=2, maxsl=n)
segments = detector.fit_predict(df)
print(pd.DataFrame(segments))
