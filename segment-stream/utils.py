import pandas as pd
import numpy as np
import cProfile, pstats, io
from pstats import SortKey


class Profiler:
    def __init__(self):
        self.pr = cProfile.Profile()
        pass

    def start(self):
        self.pr.enable()

    def stop(self):
        self.pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


def first_derivative(x: np.ndarray) -> np.ndarray:
    n = x.size
    first_derivative = np.zeros(n)
    for i in range(1, n):
        first_derivative[i] = x[i] - x[i - 1]
    return first_derivative


def second_derivative(x: np.ndarray) -> np.ndarray:
    n = x.size
    second_derivative = np.zeros(n)
    for i in range(1, n - 1):
        second_derivative[i] = x[i + 1] + x[i - 1] - 2 * x[i]
    return second_derivative


def signed_curvature(x: np.ndarray) -> np.ndarray:
    return second_derivative(x) / (1 + first_derivative(x) ** 2) ** (3 / 2)


def curvature(x: np.ndarray) -> np.ndarray:
    return signed_curvature(x).abs()


def dyadic_grid(alpha, max):
    n = np.floor(np.log(max) / np.log(1 + alpha))
    grid = np.rint((1 + alpha) ** np.arange(n))
    return np.unique(grid)


def listseries_to_df(series: pd.Series, columns=None):
    if columns is None:
        return pd.DataFrame(series.tolist(), index=series.index)
    else:
        return pd.DataFrame(series.tolist(), index=series.index, columns=columns)


def separate_lower_upper(df: pd.DataFrame):
    list_of_dfs = [
        listseries_to_df(df[column], ["ci_lower", "ci_upper"]) for column in df
    ]
    ci_lower_series = [df["ci_lower"] for df in list_of_dfs]
    ci_upper_series = [df["ci_upper"] for df in list_of_dfs]
    ci_lower = pd.concat(ci_lower_series, axis=1, keys=df.columns)
    ci_upper = pd.concat(ci_upper_series, axis=1, keys=df.columns)
    return ci_lower, ci_upper
