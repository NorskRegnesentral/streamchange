import numpy as np
import plotly.express as px
from streamchange.data import simulate
from streamchange.sequential import (
    LordenPollakScore,
    AggregatedScore,
    PenalisedScore,
    SequentialChangeDetector,
)

# Univariate score
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30])
score = LordenPollakScore(rho=4)
score.fit(x)
px.scatter(score.values_)

# Multivariate score
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30], p=5)
score = PenalisedScore(
    AggregatedScore(LordenPollakScore(rho=4), aggregator=np.sum),
    penalty=100,
)
score.fit(x)
px.scatter(score.values_)
px.scatter(x.melt(ignore_index=False), color="variable", y="value")

# Change detection
x = simulate([0, 5, 0, 20], seg_lens=[100, 30, 100, 30], p=10)
score = AggregatedScore(LordenPollakScore(rho=2), aggregator=np.sum).penalise(100)
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

# Penalty tuning
from streamchange.tuners import GridPenaltyTuner
import numpy as np

x = simulate([0, 10, 0], [1000, 100, 1000], p=1)[0]
score = AggregatedScore(LordenPollakScore(rho=0.01), aggregator=np.sum).penalise(1)
detector = SequentialChangeDetector(score, reset_on_change=True, restart_delay=100)

penalty_scales = np.geomspace(1e-6, 1000, 100)
detector = GridPenaltyTuner(detector, 1, penalty_scales)
detector.fit(x)
detector.show()

alarms = detector.predict()
fig = px.scatter(x)
for alarm in alarms:
    fig.add_vline(alarm, line_color="red")
fig.show()

# Penalty tuning tailored to sequential scores
from streamchange.sequential import SequentialScorePenaltyTuner

x = simulate([0, 10, 0], [1000, 100, 1000], p=1)[0]
score = AggregatedScore(LordenPollakScore(rho=0.01), aggregator=np.sum).penalise(1)
detector = SequentialChangeDetector(score, reset_on_change=True, restart_delay=100)

detector = SequentialScorePenaltyTuner(detector, 5, score_value_margin=0)
detector.fit(x)
detector.show()

fig = px.line(detector.detector_.penalised_scores_ + detector.detector_.get_penalty()())
fig.add_hline(detector.detector_.get_penalty()())
fig.show()
