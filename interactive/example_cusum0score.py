import numpy as np
import pandas as pd
import plotly.express as px
from streamchange.data import simulate
from streamchange.sequential.scores import CUSUM0Score
from streamchange.sequential import (
    AggregatedScore,
    SequentialChangeDetector,
)

# Univariate score
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30])[0]
score = CUSUM0Score(window_sizes=[1, 2, 5, 10])
score.fit(x)
px.scatter(x)
px.scatter(score.values_)

x = simulate([0], seg_lens=[100000])[0]
score = CUSUM0Score(window_sizes=[10, 100, 1000, 10000])
score.fit(x)


# Multivariate score
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30], p=5)
window_sizes = [2, 5, 10, 20, 50]
base_score = CUSUM0Score(window_sizes)
score = AggregatedScore(base_score, aggregator=np.sum)
penalised_score = score.penalise(100)
penalised_score.fit(x)
px.scatter(penalised_score.values_)
px.scatter(x.melt(ignore_index=False), color="variable", y="value")

# Change detection
x = simulate([0, -5, 0, 20], seg_lens=[100, 30, 100, 30], p=10)
window_sizes = [2, 5, 10, 20]
base_score = CUSUM0Score(window_sizes)
score = AggregatedScore(base_score, aggregator=sum).penalise(100)
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
window_sizes = [2000, 4000, 6000, 8000, 10000]
base_score = CUSUM0Score(window_sizes)
score = AggregatedScore(base_score, aggregator=sum)
penalised_score = score.penalise(100)
detector = SequentialChangeDetector(
    penalised_score, reset_on_change=True, restart_delay=400
)
profiler = Profiler()
profiler.start()
detector.fit(x)
profiler.stop()


# Offline score
from streamchange.offline.cusum0_score import fit_cusum0_score, nb_sum

n = 1000000
x = simulate([0], seg_lens=[n], p=6)
# window_sizes = np.arange(2, 100)
window_sizes = np.array([2000, 4000, 6000, 8000, 10000])

offline_scores = fit_cusum0_score(x.values, window_sizes, nb_sum)
offline_scores = pd.Series(offline_scores, index=x.index)

base_score = CUSUM0Score(window_sizes.tolist())
score = AggregatedScore(base_score, aggregator=sum)
online_scores = score.fit(x).values_

pd.concat([offline_scores, online_scores], axis=1)

# Offline detector
from streamchange.offline.cusum0_score import fit_cusum0_detector

x = simulate([0, -5, 0, 20], seg_lens=[100, 30, 100, 30], p=10)

window_sizes = np.array([2, 5, 10, 20])
penalty = 100
restart_delay = 20

alarms, scores = fit_cusum0_detector(
    x.values, penalty, window_sizes, restart_delay=restart_delay
)

base_score = CUSUM0Score(window_sizes.tolist())
score = AggregatedScore(base_score, aggregator=sum).penalise(penalty)
detector = SequentialChangeDetector(
    score, reset_on_change=True, restart_delay=restart_delay
)
detector.fit(x)
print(detector.penalised_scores_)
print(detector.alarms_)

pd.concat([pd.Series(scores, index=x.index), detector.penalised_scores_], axis=1).plot(
    backend="plotly"
)
