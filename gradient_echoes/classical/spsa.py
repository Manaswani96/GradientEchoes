from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable
import random
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

from ..core.objective import Objective
from ..core.callbacks import Callback

class SPSA:
    """Simultaneous Perturbation Stochastic Approximation.
    One of the best choices when gradients are noisy/expensive (quantum!).
    Works with scalars; for vectors uses NumPy if available.
    """
    def __init__(self, a=0.1, c=0.1, alpha=0.602, gamma=0.101):
        self.a, self.c, self.alpha, self.gamma = a, c, alpha, gamma

    def _rand_delta(self, x):
        if np is not None and hasattr(x, "__len__"):
            # Rademacher ±1
            return np.where(np.random.rand(*np.shape(x)) < 0.5, -1.0, 1.0)
        return -1.0 if random.random() < 0.5 else 1.0

    def _scale(self, s, x):
        if np is not None and hasattr(x, "__len__"): return s * np.ones_like(x)
        return s

    def minimize(self, obj: Objective, steps: int = 200, callback: Optional[Callback] = None):
        x = obj.project(obj.init)
        history: List[Dict[str, Any]] = []
        for k in range(1, steps + 1):
            ak = self.a / (k ** self.alpha)
            ck = self.c / (k ** self.gamma)
            delta = self._rand_delta(x)
            x_plus  = obj.project(x + self._scale(ck, delta))
            x_minus = obj.project(x - self._scale(ck, delta))
            f_plus, f_minus = obj.f(x_plus), obj.f(x_minus)
            gk = (f_plus - f_minus) / (2 * ck) * (1.0 / delta)  # elementwise * (1/delta)
            x = obj.project(x - self._scale(ak, gk))
            f_curr = obj.f(x)
            row = {"f": float(f_curr), "grad_norm2": 0.0, "lr": float(ak)}
            history.append(row)
            if callback: callback(k, x, row)
        return x, history
