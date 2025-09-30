from __future__ import annotations
from typing import Iterable, Callable, Any, Tuple, List
from math import pi

def parameter_shift_grad(expectation: Callable[[Any], float], theta: Any, s: float = pi/2):
    """Parameter-shift rule (finite-shift surrogate).
    - If theta is scalar: returns scalar grad.
    - If theta is iterable: returns tuple of grads (one param shifted at a time).
    """
    try:
        it = list(theta)  # treat as vector-like
        grads: List[float] = []
        for i in range(len(it)):
            tp = it.copy(); tm = it.copy()
            tp[i] = tp[i] + s; tm[i] = tm[i] - s
            grads.append(0.5 * (expectation(tp) - expectation(tm)))
        return tuple(grads)
    except TypeError:
        # scalar parameter
        return 0.5 * (expectation(theta + s) - expectation(theta - s))
