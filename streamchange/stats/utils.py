import pandas as pd


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
