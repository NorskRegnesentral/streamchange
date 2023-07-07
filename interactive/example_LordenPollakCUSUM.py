import plotly.express as px

from streamchange.sequential import LordenPollakCUSUM
from streamchange.data import simulate

x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30])[0]
detector = LordenPollakCUSUM(rho=4, penalty=100, restart_delay=10)
score = []
res = []
for t, x_t in x.items():
    detector.update(x_t)
    score.append(detector.score)
    if detector.change_detected:
        mean = detector.sum / detector.n
        res.append(dict(t=t, cpt=detector.changepoints, mean=mean))

fig = px.scatter(x=range(len(score)), y=score, render_mode="webgl")
fig.add_hline(detector.penalty(), line_color="red")
fig.show()

# Using .fit_predict()
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30])[0]
detector = LordenPollakCUSUM(rho=4, penalty=100, restart_delay=30)
cpts = detector.fit_predict(x)
fig = px.scatter(x)
for cpt in cpts:
    fig.add_vline(cpt, line_color="red")
fig.show()


# Penalty tuning
from streamchange.tuners import OptunaPenaltyTuner
import numpy as np

x = simulate([0, 10, 0], [1000, 100, 1000], p=1)[0]
detector = LordenPollakCUSUM(rho=0.0001, penalty=1, restart_delay=100)

penalty_scales = np.geomspace(1e-6, 1000, 100)
tuner = OptunaPenaltyTuner(detector, 100, penalty_scales)
tuner.fit(x)
tuner.show()

cpts = detector.fit_predict(x)
fig = px.scatter(x)
for cpt in cpts:
    fig.add_vline(cpt, line_color="red")
fig.show()
