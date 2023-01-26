import plotly.express as px
from river.stream import iter_pandas

from streamchange.detector import LordenPollakCUSUM
from streamchange.data import simulate

df = simulate([0, 5], seg_lens=[100, 10])

detector = LordenPollakCUSUM(rho=4, threshold=100)
score = []
res = []
for t, (x, _) in enumerate(iter_pandas(df)):
    detector.update(x)
    score.append(detector.score)
    if detector.change_detected:
        mean = detector.sum / detector.n
        res.append(dict(t=t, cpt=detector.changepoints, mean=mean))

fig = px.scatter(x=range(len(score)), y=score, render_mode="webgl")
fig.add_hline(detector.threshold, line_color="red")
fig.show()
