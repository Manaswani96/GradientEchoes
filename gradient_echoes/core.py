# gradient_echoes/core.py
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np

@dataclass
class Result:
    x_best: np.ndarray
    best_value: float
    nit: int
    history: List[Tuple[int, float]]  # (iter, value)

def minimize(optimizer, func: Callable, x0, *, grad: Optional[Callable]=None,
             max_iters: Optional[int]=None, callback: Optional[Callable]=None, seed: Optional[int]=None) -> Result:
    """
    Simple wrapper that calls an optimizer's minimize method.
    The optimizer should implement: minimize(func, x0, grad=..., max_iters=..., callback=..., seed=...)
    """
    return optimizer.minimize(func, x0, grad=grad, max_iters=max_iters, callback=callback, seed=seed)
