# streamchange
A package for segmenting streaming time series data into homogenous segments. The segmentation is based on statistical change-point detection (aka online/sequential/iterative change-point detection). Inspired by [river](https://riverml.xyz/0.14.0/) and follows its API.


## Quickstart
```python
>>> from river.stream import iter_pandas
>>> 
>>> from streamchange.amoc_test import UnivariateCUSUM
>>> from streamchange.detector import WindowSegmentor, JumpbackWindow
>>> from streamchange.data import simulate
>>>
>>> df = simulate([0, 10, 0], [100], p=1)
>>> test = UnivariateCUSUM(minsl=1).set_default_threshold(10 * df.size)
>>> window = JumpbackWindow(4, 100)
>>> detector = WindowSegmentor(test, window)
>>> cpts = []
>>> for t, (x, _) in enumerate(iter_pandas(df)):
...     detector.update(x)
...     if detector.change_detected:
...         cpts.append((t, detector.changepoints))
print(cpts)
[(100, [-2]), (200, [-2])]
```
Throughout this package, a change-point is defined as the negative index from
the time of detection.

## Installation
```sh
pip install git+https://github.com/NorskRegnesentral/streamchange
```

## Dependencies
- `pandas` >= 1.5
- `numpy` >= 1.23
- `numba` >= 0.56
- `river` >=0.14

You also need Python >= 3.8. 

## License

Streamchange is a free and open-source software licensed under the [BSD 3-clause license](https://github.com/NorskRegnesentral/streamchange/blob/main/LICENSE).
