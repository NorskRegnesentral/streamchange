from streamchange.detector import UnivariateCUSUM, SegmentorCollection
from streamchange.penalty_tuner import SimpleTuner
from streamchange.segment_summarizer import separate_lower_upper
from streamchange.utils.example_data import three_segments_data
from streamchange.plot import MultivariateTimeSeriesFigure

import numpy as np

# Read data
seg_len = 1000
df = three_segments_data(p=2, seg_len=seg_len, mean_change=2)
df.iloc[[2, df.shape[0] - 2], 0] = None
df_train = df.iloc[: 2 * seg_len]
df_test = df.iloc[2 * seg_len :]

# Method setup and tuning
confidence_level = 0.99
alpha = (1 - confidence_level) / 2
estimators = {
    "mean": np.average,
    "confidence_interval": lambda x: np.quantile(x, [alpha, 1 - alpha]),
}
base_segmentor = UnivariateCUSUM(
    min_size_window=4, max_size_window=50, max_size_history=1000, estimators=estimators
)
segmentor = SegmentorCollection(df.columns, base_segmentor)
tune = SimpleTuner(alpha=0.5, max_cpts=10, sampling_probability=0.1)
tune(segmentor, df_train)

# Segmentation/model fitting
segmentor = segmentor.update_fit(df_train)
for timestamp, row in df_test.to_dict(orient="index").items():
    segmentor.update_fit_dict_int(row, timestamp)
    segmentor.model_now("confidence_interval")

# Plot results
model = segmentor.wide_model_at(df.index.tolist())
n_segments = len(segmentor.changepoint_timestamps())
fig = MultivariateTimeSeriesFigure(df.columns, title=f"{n_segments} segments")
fig.add_raw_data(df)
fig.add_mean(model.segment_mean)
fig.add_confidence_band(*separate_lower_upper(model.segment_confidence_interval))
fig.show()
