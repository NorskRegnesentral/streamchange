from streamchange.detector import UnivariateCUSUM
from streamchange.threshold_tuner import SimpleTuner
from streamchange.utils.example_data import three_segments_data
from streamchange.plot import TimeSeriesFigure

seg_len = 10000
df = three_segments_data(p=2, seg_len=seg_len, mean_change=2)[0]
df_train = df.loc[0 : (2 * seg_len - 1)]
df_test = df.loc[(2 * seg_len) : (3 * seg_len)]

# Set up the segmentor and tune it.
segmentor = UnivariateCUSUM(
    min_size_window=4, max_size_window=50, max_size_history=1000
)
tune = SimpleTuner(0.5, max_cpts=500, sampling_probability=0.1)
tune(segmentor, df_train)
tune.show()

# Run segmentation on the training and continuously update on test set.
segmentor.update_fit(df_train)
for index, value in df_test.items():
    segmentor.update_fit(value, index)

# Plot results
model = segmentor.model_at(df.index.tolist())
n_segments = model.segment_start.nunique()
fig = TimeSeriesFigure(title=f"{n_segments} segments")
fig.add_raw_data(df)
fig.add_mean(model.segment_mean)
ci_lower = model.segment_mean - 1.96 * model.segment_sd
ci_upper = model.segment_mean + 1.96 * model.segment_sd
fig.add_confidence_band(ci_lower, ci_upper)
fig.show()
