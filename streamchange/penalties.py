import abc
import numpy as np


class BasePenalty:
    def __init__(self, scale: float = 1.0):
        if scale < 0:
            raise ValueError("scale must be >= 0.")
        self.scale = scale

    @abc.abstractmethod
    def default_penalty(self, affected_size: int) -> float:
        pass

    def __call__(self, affected_size: int = 1) -> float:
        return self.scale * self.default_penalty(affected_size)


class ConstantPenalty(BasePenalty):
    def __init__(self, value, scale=1.0):
        super().__init__(scale)
        if value < 0:
            raise ValueError("The ConstantPenalty value must be >= 0.")
        self.value = value

    def default_penalty(self, affected_size=None):
        return self.value


class BIC(ConstantPenalty):
    def __init__(self, arl: int = 10000, p: int = 1, scale=1.0):
        self.arl = arl
        self.p = p
        value = 2 * self.p * np.log(self.arl)
        super().__init__(value, scale)


class LinearPenalty(BasePenalty):
    def __init__(self, intercept, slope, scale=1.0):
        super().__init__(scale)
        if intercept < 0:
            raise ValueError("The LinearPenalty intercept must be >= 0.")
        if slope < 0:
            raise ValueError("The LinearPenalty slope must be >= 0.")
        self.intercept = intercept
        self.slope = slope

    def default_penalty(self, affected_size):
        return self.intercept + affected_size * self.slope


class LinearConstPenalty(BasePenalty):
    def __init__(
        self, constant_value, intercept, slope, transition_point=None, scale=1.0
    ):
        super().__init__(scale)
        self.constant_value = constant_value
        self.intercept = intercept
        self.slope = slope
        self._constant_penalty = ConstantPenalty(constant_value)
        self._linear_penalty = LinearPenalty(intercept, slope)
        self.transition_point = (
            ((constant_value - intercept) / slope if slope > 0 else 0)
            if transition_point is None
            else transition_point
        )

    def default_penalty(self, affected_size):
        if affected_size <= self.transition_point:
            return self._linear_penalty(affected_size)
        else:
            return self._constant_penalty()
