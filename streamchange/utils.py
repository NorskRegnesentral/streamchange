import numpy as np
import cProfile, pstats, io
from pstats import SortKey


def geomspace_int(start: int, stop: int, step: float = 2.0) -> np.ndarray:
    if step <= 1.0:
        raise ValueError(f"Step must be > 1.0, but step={step}.")
    if stop < start:
        raise ValueError(f"Stop is smaller than start: stop={stop}, start={start}.")

    values = [start]
    while values[-1] * step < stop:
        next_value = int(np.ceil(values[-1] * step))
        values.append(next_value)

    if values[-1] < stop:
        values.append(stop)

    return np.array(values)


def has_method(obj, method_name):
    method = getattr(obj, method_name, None)
    return callable(method)


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
