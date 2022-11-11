import numpy as np

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

