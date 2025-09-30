from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable
from math import sqrt
from ..core.objective import Objective
from ..core.oracle import Oracle
from ..core.callbacks import Callback
from ..core.schedules import Constant
from ..mathops import sub, mul

class AMSGrad:
    """AMSGrad: Adam variant with non-increasing second-moment estimate."""
    def __init__(self, lr: Callable[[int], float] | float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr if callable(lr) else Constant(lr)
        self.b1, self.b2, self.eps = float(beta1), float(beta2), float(eps)

    def minimize(self, obj: Objective, oracle: Oracle | None = None, steps: int = 200, callback: Optional[Callback] = None):
        if oracle is None: oracle = Oracle(obj.f, obj.grad)
        x = obj.project(obj.init)
        m = 0.0; v = 0.0; vhat_max = 0.0
        history: List[Dict[str, Any]] = []
        for t in range(1, steps + 1):
            lr = float(self.lr(t))
            f, g, extra = oracle(x)
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * (g * g)
            mhat = m / (1 - self.b1 ** t)
            vhat = v / (1 - self.b2 ** t)
            vhat_max = max(vhat_max, vhat)
            x = obj.project(sub(x, mul(lr, mhat / (sqrt(vhat_max) + self.eps))))
            row = {"f": float(f), "grad_norm2": float(extra["grad_norm2"]), "lr": lr}
            history.append(row)
            if callback: callback(t, x, row)
        return x, history
