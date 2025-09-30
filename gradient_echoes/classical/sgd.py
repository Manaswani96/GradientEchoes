from __future__ import annotations
from typing import Optional, List, Dict, Any, Callable
from ..core.objective import Objective
from ..core.oracle import Oracle, StochasticOracle
from ..core.callbacks import Callback
from ..core.schedules import Constant
from ..mathops import sub, mul
# SGD with optional momentum and Nesterov

class SGD:
    def __init__(
        self,
        lr: Callable[[int], float] | float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
    ):
        self.lr = lr if callable(lr) else Constant(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

    def minimize(
        self,
        obj: Objective,
        oracle: Oracle | StochasticOracle | None = None,
        steps: int = 200,
        callback: Optional[Callback] = None,
        key: Optional[int] = None,
    ):
        if oracle is None:
            oracle = Oracle(obj.f, obj.grad)

        x = obj.project(obj.init)
        v = 0.0  # momentum buffer (scalar or vector via mathops.mul/add)
        history: List[Dict[str, Any]] = []

        for t in range(steps):
            lr = float(self.lr(t))
            if self.nesterov and self.momentum:
                x_look = obj.project(sub(x, mul(self.momentum, v)))
                f, g, extra = oracle(x_look) if isinstance(oracle, StochasticOracle) else oracle(x_look)
            else:
                f, g, extra = oracle(x) if isinstance(oracle, StochasticOracle) else oracle(x)

            v = mul(self.momentum, v) + mul(lr, g)
            x = obj.project(sub(x, v))
            row = {"f": float(f), "grad_norm2": float(extra["grad_norm2"]), "lr": lr}
            history.append(row)
            if callback: callback(t, x, row)
        return x, history
