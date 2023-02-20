import numpy as np
from streamchange.data import simulate
from streamchange.capa import Capa, ConstMeanL2

df = simulate([0, 10, 0], [1000])[0]
df.iloc[10] = 200
csaving = ConstMeanL2()
capa = Capa(csaving, minsl=2, maxsl=1000)
collective_anoms, point_anoms = capa.fit(df).predict()
print(collective_anoms)
print(point_anoms)
