from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable
from ..core.objective import Objective
from ..core.oracle import Oracle
from ..core.callbacks import Callback
from ..core.schedules import Constant
from ..mathops import sub, mul

class AdaGrad:
    """AdaGrad: accumulates squared gradients; great for sparse features."""
    def __init__(self, lr: Callable[[int], float] | float = 1e-2, eps: float = 1e-8):
        self.lr = lr if callable(lr) else Constant(lr)
        self.eps = float(eps)

    def minimize(self, obj: Objective, oracle: Oracle | None = None, steps: int = 200, callback: Optional[Callback] = None):
        if oracle is None: oracle = Oracle(obj.f, obj.grad)
        x = obj.project(obj.init)
        G = 0.0
        history: List[Dict[str, Any]] = []
        for t in range(1, steps + 1):
            lr = float(self.lr(t))
            f, g, extra = oracle(x)
            G += g * g
            x = obj.project(sub(x, mul(lr / ((G ** 0.5) + self.eps), g)))
            row = {"f": float(f), "grad_norm2": float(extra["grad_norm2"]), "lr": lr}
            history.append(row)
            if callback: callback(t, x, row)
        return x, history
