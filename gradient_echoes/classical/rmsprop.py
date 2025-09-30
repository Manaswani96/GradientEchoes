from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable
from ..core.objective import Objective
from ..core.oracle import Oracle
from ..core.callbacks import Callback
from ..core.schedules import Constant
from ..mathops import sub, mul

class RMSProp:
    """RMSProp (with optional 'centered' variant).
    Good when gradients are noisy; adaptively scales by running RMS of grad.
    """
    def __init__(
        self,
        lr: Callable[[int], float] | float = 1e-3,
        alpha: float = 0.99,
        eps: float = 1e-8,
        centered: bool = False,
    ):
        self.lr = lr if callable(lr) else Constant(lr)
        self.alpha, self.eps, self.centered = float(alpha), float(eps), bool(centered)

    def minimize(
        self,
        obj: Objective,
        oracle: Oracle | None = None,
        steps: int = 200,
        callback: Optional[Callback] = None,
    ):
        if oracle is None: oracle = Oracle(obj.f, obj.grad)
        x = obj.project(obj.init)
        avg_sq = 0.0
        avg_g = 0.0
        history: List[Dict[str, Any]] = []
        for t in range(1, steps + 1):
            lr = float(self.lr(t))
            f, g, extra = oracle(x)
            avg_sq = self.alpha * avg_sq + (1 - self.alpha) * (g * g)
            if self.centered:
                avg_g = self.alpha * avg_g + (1 - self.alpha) * g
                denom = (avg_sq - avg_g * avg_g + self.eps) ** 0.5
            else:
                denom = (avg_sq + self.eps) ** 0.5
            step = g / denom
            x = obj.project(sub(x, mul(lr, step)))
            row = {"f": float(f), "grad_norm2": float(extra["grad_norm2"]), "lr": lr}
            history.append(row)
            if callback: callback(t, x, row)
        return x, history
