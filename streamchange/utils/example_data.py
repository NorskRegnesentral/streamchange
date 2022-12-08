import pandas as pd
import numpy as np


def outlier_data(n=1000, p=2, n_outliers=4, outlier_size=20, seed=10):
    np.random.seed(seed)
    mean = np.zeros(p)
    cov = np.identity(p)
    x = np.random.multivariate_normal(mean, cov, n)
    outlier_positions = np.linspace(0, n - 1, n_outliers, dtype=int)
    x[outlier_positions] = x[outlier_positions] + outlier_size
    return pd.DataFrame(x, index=range(len(x)))


def three_segments_data(p=2, seg_len=100, mean_change=2, seed=10):
    np.random.seed(seed)
    base_mean = np.zeros(p)
    change_mean = base_mean + mean_change
    cov = np.identity(p)
    x = np.concatenate(
        (
            np.random.multivariate_normal(base_mean, cov, seg_len),
            np.random.multivariate_normal(change_mean, cov, seg_len),
            np.random.multivariate_normal(base_mean, cov, seg_len),
        )
    )
    return pd.DataFrame(x, index=range(len(x)))


def simulate_changing_data(means, seg_lens=[100], p=1, seed=10):
    np.random.seed(seed)
    cov = np.identity(p)
    if len(seg_lens) == 1:
        seg_lens = list(np.repeat(seg_lens[0], len(means)))

    segments = [
        np.random.multivariate_normal(np.repeat(mean, p), cov, seg_len)
        for mean, seg_len in zip(means, seg_lens)
    ]
    x = np.concatenate(tuple(segments))
    return pd.DataFrame(x, index=range(len(x)))
