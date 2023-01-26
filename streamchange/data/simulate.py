import pandas as pd
import numpy as np


def simulate(means, seg_lens=[100], p=1, n_outliers=0, outlier_size=0, seed=10):
    np.random.seed(seed)
    cov = np.identity(p)
    if len(seg_lens) == 1:
        seg_lens = list(np.repeat(seg_lens[0], len(means)))

    segments = [
        np.random.multivariate_normal(np.repeat(mean, p), cov, seg_len)
        for mean, seg_len in zip(means, seg_lens)
    ]
    x = np.concatenate(tuple(segments))
    outlier_positions = np.linspace(0, x.size - 1, n_outliers, dtype=int)
    x[outlier_positions] = x[outlier_positions] + outlier_size
    return pd.DataFrame(x, index=range(len(x)))
