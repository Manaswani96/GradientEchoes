import numpy as np
from typing import Callable, Optional
from ..core import Result

try:
    from scipy.optimize import minimize as sp_minimize
except Exception:
    sp_minimize = None

class BFGSWrapper:
    """Thin wrapper around scipy.optimize.minimize(method='BFGS').
    If scipy is not installed, user will be prompted to install it.
    """
    def __init__(self, options=None):
        self.options = options or {}

    def minimize(self, func: Callable, x0, *, grad: Optional[Callable]=None, max_iters: Optional[int]=None,
                 callback: Optional[Callable]=None, seed: Optional[int]=None):
        if sp_minimize is None:
            raise ImportError("scipy not found. Install `scipy` (or use extras) to use BFGSWrapper.")
        res = sp_minimize(fun=func, x0=np.asarray(x0, dtype=float), jac=grad, method='BFGS', options=self.options)
        history = []  # we don't capture per-iter history from scipy here
        return Result(x_best=res.x, best_value=float(res.fun), nit=int(res.nit) if hasattr(res, 'nit') else 0, history=history)
