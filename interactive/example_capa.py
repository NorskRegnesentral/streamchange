import numpy as np
from streamchange.data import simulate
from streamchange.capa import Capa, ConstMeanL2

df = simulate([0, 10, 0], [100])
df.iloc[10] = 200
capa = Capa(ConstMeanL2(), minsl=2, maxsl=1000, predict_point_anomalies=True)
anomalies = capa.fit_predict(df)
print(anomalies)
print(capa.collective_anomalies_)
print(capa.point_anomalies_)
