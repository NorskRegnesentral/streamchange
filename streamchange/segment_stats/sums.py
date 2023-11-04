from collections import deque
from numbers import Number


class MovingSum:
    """Moving sum of the last `window_size` elements.

    Approximately 5-7x times faster than river.utils.Rolling(Sum(), window_size=window_size)
    Worth having specialised implementations for common use cases.

    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.reset()

    def reset(self):
        self.window.clear()
        self._sum = 0.0
        return self

    @property
    def value(self):
        return self._sum

    def update(self, x: Number):
        if len(self.window) == self.window_size:
            self._sum -= self.window.popleft()
        self._sum += x
        self.window.append(x)
        return self
