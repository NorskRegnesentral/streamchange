# streamchange
A package for segmenting streaming time series data into homogenous segments. The segmentation is based on statistical change-point detection (aka online/sequential/iterative change-point detection). Inspired by [river](https://riverml.xyz/0.14.0/) and follows its API.

# OBS
The functionality in this package is about to be moved into `skchange`: https://github.com/NorskRegnesentral/skchange.

## Quickstart
```python
>>> from river.stream import iter_pandas
>>> 
>>> from streamchange.amoc import UnivariateCUSUM, WindowSegmentor
>>> from streamchange.data import simulate
>>>
>>> df = simulate([0, 10, 0], [100], p=1)
>>> test = UnivariateCUSUM(minsl=1).set_default_threshold(10 * df.size)
>>> detector = WindowSegmentor(test, min_window=4, max_window=100)
>>> cpts = []
>>> for t, (x, _) in enumerate(iter_pandas(df)):
...     detector.update(x)
...     if detector.change_detected:
...         cpts.append((t, detector.changepoints))
print(cpts)
[(100, [1]), (200, [1])]
```
Throughout this package, a change-point is defined as the end of a segment, 
and it is stored relative to the time of detection by its index backwards in time.
For example, if t is the index of the current observation, the changepoint in the
external data set is given by t - changepoint.
In the example above, 100 and 200 are the times of detection, and the changepoints
are located at indices 99 and 199.

## Installation
```sh
pip install git+https://github.com/NorskRegnesentral/streamchange
```

## Dependencies
- `pandas` >= 1.3
- `numpy` >= 1.19
- `numba` >= 0.56

You also need Python >= 3.8. 

Optional dependencies:
- Online estimation of segment parameters: `river` >= 0.14
- Penalty tuning: `optuna` >= 3.1.1


## License

Streamchange is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/streamchange/blob/main/LICENSE).
