import numpy as np
import pandas as pd
from streamchange.data import simulate
from streamchange.pelt import Pelt, L2Cost

df = simulate([0, 10, 0, 10, 0], [1000])
detector = Pelt(L2Cost(), minsl=2, maxsl=10000)
segments = detector.fit_predict(df)
print(pd.DataFrame(segments))
