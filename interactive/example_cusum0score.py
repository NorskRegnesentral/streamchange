import numpy as np
import plotly.express as px
from streamchange.data import simulate
from streamchange.sequential.scores import CUSUM0Score
from streamchange.sequential import (
    AggregatedScore,
    SequentialChangeDetector,
)

# Univariate score
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30])[0]
score = CUSUM0Score(candidate_grid=[1, 2, 5, 10])
score.fit(x)
px.scatter(x)
px.scatter(score.values_)

x = simulate([0], seg_lens=[100000])[0]
score = CUSUM0Score(candidate_grid=[10, 100, 1000, 10000])
score.fit(x)


# Multivariate score
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30], p=5)
candidate_grid = [2, 5, 10, 20, 50]
base_score = CUSUM0Score(candidate_grid)
score = AggregatedScore(base_score, aggregator=np.sum)
penalised_score = score.penalise(100)
penalised_score.fit(x)
px.scatter(penalised_score.values_)
px.scatter(x.melt(ignore_index=False), color="variable", y="value")

# Change detection
x = simulate([0, -5, 0, 20], seg_lens=[100, 30, 100, 30], p=10)
candidate_grid = [2, 5, 10, 20]
base_score = CUSUM0Score(candidate_grid)
score = AggregatedScore(base_score, aggregator=np.sum).penalise(100)
detector = SequentialChangeDetector(score, reset_on_change=True, restart_delay=50)
detector.fit(x)
# fig = px.scatter(detector.penalised_scores_)
fig = px.scatter(x.melt(ignore_index=False), color="variable", y="value")
for alarm in detector.alarms_:
    fig.add_vline(alarm, line_color="red")
for cpt in detector.changepoints_:
    fig.add_vline(cpt, line_color="blue")
fig.show()
px.scatter(detector.penalised_scores_)

from streamchange.utils import Profiler

# Multivariate score, timing
x = simulate([0], seg_lens=[10000], p=6)
candidate_grid = [2000, 4000, 6000, 8000, 10000]
base_score = CUSUM0Score(candidate_grid)
score = AggregatedScore(base_score, aggregator=sum)
penalised_score = score.penalise(100)
detector = SequentialChangeDetector(
    penalised_score, reset_on_change=True, restart_delay=400
)
profiler = Profiler()
profiler.start()
detector.fit(x)
profiler.stop()
