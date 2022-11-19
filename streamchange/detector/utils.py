import numpy as np


def get_public_properties(object):
    return {
        name: attr
        for name, attr in object.__dict__.items()
        if not name.startswith("_") and not callable(attr)
    }


def dyadic_grid(alpha, max):
    n = np.floor(np.log(max) / np.log(1 + alpha))
    grid = np.rint((1 + alpha) ** np.arange(n))
    return np.unique(grid)
