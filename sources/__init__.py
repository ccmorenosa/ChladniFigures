"""Set of possible source for the plate system."""
import numpy as np


def sine_source(t, w):
    """Sinusoidal signal."""
    return np.sin(w * t)


def square_source(t, w):
    """Sawtooth signal."""
    # Define the period.
    T = 1 / w

    # Get time in the range [0,T).
    t %= T

    # Evaluate cases.
    if t <= T/2:
        return 1

    elif t <= T:
        return -1


def triangular_source(t, w):
    """Triangular signal."""
    # Define the period.
    T = 1 / w

    # Get time in the range [0,T).
    t %= T

    # Evaluate cases.
    if t <= T/4:
        return t * 4/T

    elif t <= 3*T/4:
        return - t * 4/T + 2

    elif t <= T:
        return t * 4/T - 4


def sawtooth_source(t, w):
    """Sawtooth signal."""
    # Define the period.
    T = 1 / w

    # Get time in the range [0,T).
    t %= T

    return t * 1/T - 1


sources_functions = {
    "sine": sine_source,
    "square": square_source,
    "triangular": triangular_source,
    "sawtooth": sawtooth_source
}
