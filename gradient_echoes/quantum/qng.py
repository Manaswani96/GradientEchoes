from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable
from ..core.objective import Objective
from ..core.oracle import Oracle
from ..core.callbacks import Callback
from ..core.schedules import Constant
from ..mathops import sub, mul

class QNG:
    """Quantum Natural Gradient (interface-ready).
    We accept a 'metric' callback via obj.manifold.metric(x, v)->v^T G(x) v.
    If metric is None, we gracefully fall back to SGD.
    """
    def __init__(self, lr: Callable[[int], float] | float = 0.1):
        self.lr = lr if callable(lr) else Constant(lr)

    def minimize(self, obj: Objective, oracle: Oracle | None = None, steps: int = 100, callback: Optional[Callback] = None):
        if oracle is None: oracle = Oracle(obj.f, obj.grad)
        x = obj.project(obj.init)
        history: List[Dict[str, Any]] = []

        for t in range(steps):
            lr = float(self.lr(t))
            f, g, extra = oracle(x)

            # If a metric is provided, precondition the gradient by (approx) G^{-1} g.
            # Here we just scale by 1 / sqrt(metric(x,g)+eps) as a cheap stand-in.
            if obj.manifold.metric is not None:
                m = obj.manifold.metric(x, g) + 1e-12
                pre_g = mul(1.0 / (m ** 0.5), g)
            else:
                pre_g = g

            x = obj.project(sub(x, mul(lr, pre_g)))
            row = {"f": float(f), "grad_norm2": float(extra["grad_norm2"]), "lr": lr}
            history.append(row)
            if callback: callback(t, x, row)
        return x, history
