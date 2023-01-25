import plotly.express as px

from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import LordenPollakCUSUM
from streamchange.utils.example_data import three_segments_data

seg_len = 1000
df = three_segments_data(p=1, seg_len=seg_len, mean_change=10)[0]

detector = LordenPollakCUSUM(4, 100)
score = []
cpts = []
for t, x in df.items():
    detector.update({df.name: x})
    score.append(detector.score)
    if detector.change_detected:
        cpts.append(t)
print(cpts)

px.scatter(x=range(len(score)), y=score, render_mode="webgl")
