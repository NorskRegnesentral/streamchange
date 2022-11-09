import numpy as np


def dyadic_grid(alpha, max):
    n = np.floor(np.log(max) / np.log(1 + alpha))
    grid = np.rint((1 + alpha) ** np.arange(n))
    return np.unique(grid)
