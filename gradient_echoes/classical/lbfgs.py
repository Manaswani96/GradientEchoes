from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from ..core.objective import Objective
from ..core.oracle import Oracle
from ..core.callbacks import Callback
from ..mathops import add, sub, mul, norm2

# Minimal L-BFGS with two-loop recursion; line search is backtracking (Armijo-like).
class LBFGS:
    def __init__(self, history_size: int = 10, c1: float = 1e-4, backtrack: float = 0.5, init_scale: float = 1.0):
        self.m = int(history_size)
        self.c1 = float(c1)
        self.bt = float(backtrack)
        self.H0 = float(init_scale)

    def minimize(self, obj: Objective, oracle: Oracle | None = None, steps: int = 100, callback: Optional[Callback] = None):
        if oracle is None:
            oracle = Oracle(obj.f, obj.grad)

        x = obj.project(obj.init)
        s_list: List[Any] = []
        y_list: List[Any] = []
        rho_list: List[float] = []

        f, g, _ = oracle(x)
        history: List[Dict[str, Any]] = []

        for t in range(steps):
            # two-loop recursion
            q = g
            alpha: List[float] = []
            for i in reversed(range(len(s_list))):
                s, y, rho = s_list[i], y_list[i], rho_list[i]
                a = rho * (s * q if hasattr(s, "__mul__") else s * q)  # scalar dot for 1-D; kept simple
                alpha.append(a)
                q = sub(q, mul(a, y))
            r = mul(self.H0, q)
            for i in range(len(s_list)):
                s, y, rho = s_list[i], y_list[i], rho_list[i]
                b = rho * (y * r if hasattr(y, "__mul__") else y * r)
                r = add(r, mul((alpha[-1 - i] - b), s))
            p = mul(-1.0, r)  # search direction

            # Armijo backtracking
            step = 1.0
            gTp = (g * p if hasattr(g, "__mul__") else g * p)
            while True:
                x_new = obj.project(add(x, mul(step, p)))
                f_new, g_new, _extra = oracle(x_new)
                if f_new <= f + self.c1 * step * gTp:
                    break
                step *= self.bt
                if step < 1e-12:
                    break

            s = sub(x_new, x)
            y = sub(g_new, g)
            ys = (y * s if hasattr(y, "__mul__") else y * s)
            if ys != 0.0:
                rho = 1.0 / ys
                s_list.append(s); y_list.append(y); rho_list.append(rho)
                if len(s_list) > self.m:
                    s_list.pop(0); y_list.pop(0); rho_list.pop(0)

            x, f, g = x_new, f_new, g_new
            row = {"f": float(f), "grad_norm2": float(norm2(g)), "lr": float(step)}
            history.append(row)
            if callback: callback(t, x, row)

        return x, history
