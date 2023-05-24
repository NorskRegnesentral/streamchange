import plotly.express as px

from streamchange.sequential import LordenPollakCUSUM
from streamchange.data import simulate

series = simulate([0, 5], seg_lens=[100, 10])[0]

detector = LordenPollakCUSUM(rho=4, penalty=100)
score = []
res = []
for t, x in series.items():
    detector.update(x)
    score.append(detector.score)
    if detector.change_detected:
        mean = detector.sum / detector.n
        res.append(dict(t=t, cpt=detector.changepoints, mean=mean))

fig = px.scatter(x=range(len(score)), y=score, render_mode="webgl")
fig.add_hline(detector.penalty(), line_color="red")
fig.show()
