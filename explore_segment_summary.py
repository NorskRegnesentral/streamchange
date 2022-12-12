from streamchange.amoc_test import UnivariateCUSUM
from streamchange.detector import WindowSegmentor
from streamchange.stats import Mean
from streamchange.utils.example_data import three_segments_data

seg_len = 100000
df = three_segments_data(p=1, seg_len=seg_len, mean_change=10)[0]

##################
## Single stat ##
##################
test = UnivariateCUSUM().set_default_threshold(10 * df.size)
detector = WindowSegmentor(test, min_window=4, max_window=100)
stat = Mean()

cpts = []
segment_stats = []
for t, x in df.items():
    detector.update({df.name: x})
    stat.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
        x_since_cpt = df.iloc[t - detector.changepoints[-1] + 1 : t + 1]
        stat.revert(x_since_cpt.mean(), x_since_cpt.size)
        segment_stats.append(stat.get())
        stat.restart(x_since_cpt.to_numpy())
print(cpts)
print(segment_stats)


###################
## Several stats ##
###################
test = UnivariateCUSUM().set_default_threshold(10 * df.size)
detector = WindowSegmentor(test, min_window=4, max_window=100)
stats = {
    "mean": Mean(),
    "quantile01": Quantile(0.01),
    "quantile99": Quantile(0.99),
}

cpts = []
segment_stats = []
for t, x in df.items():
    detector.update({df.name: x})
    for stat in stats.values():
        stat.update(x)
    if detector.change_detected:
        cpts.append((t, detector.changepoints))
        x_since_cpt = df.iloc[t - detector.changepoints[-1] + 1 : t + 1]
        segment_stats.append({})
        for name, stat in stats.items():
            stat.revert(x_since_cpt.mean(), x_since_cpt.size)
            segment_stats[-1][name] = stat.get()
            stat.revert(stat.get(), stat.n)
            stat.update_many(x_since_cpt.to_numpy())
print(cpts)
print(segment_stats)

# Custom needs/utilities:
#  - stat.revert() for all stats.
#  - stat.restart(x_since_cpt): reset stat, then .update_many().
#  - Implement these functions on collection level to avoid the visible loops over stats?
#      * Similar to Transformer Collection?
#  - Pipeline: ChangeDetector -> stats?
#     * Single .update, .
#     * Similar to Transformer Pipeline?
