from __future__ import annotations
from typing import Optional, List, Dict, Any
from ..core.objective import Objective
from ..core.callbacks import Callback

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

class NelderMead:
    """Derivative-free simplex method (requires NumPy for n>1). Good for black-box & quantum sims."""
    def __init__(self, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        self.alpha, self.gamma, self.rho, self.sigma = alpha, gamma, rho, sigma

    def minimize(self, obj: Objective, steps: int = 200, callback: Optional[Callback] = None):
        if np is None:
            # 1D fallback: reflect endpoints around middle
            x_lo = obj.init - 1.0
            x_hi = obj.init + 1.0
            f_lo, f_hi = obj.f(x_lo), obj.f(x_hi)
            history: List[Dict[str, Any]] = []
            x = obj.init
            for t in range(steps):
                # take the better of the two
                if f_lo < f_hi:
                    x = x_lo
                    x_hi = x + self.alpha * (x - x_hi)
                    f_hi = obj.f(x_hi)
                else:
                    x = x_hi
                    x_lo = x + self.alpha * (x - x_lo)
                    f_lo = obj.f(x_lo)
                row = {"f": float(obj.f(x)), "grad_norm2": 0.0, "lr": 0.0}
                history.append(row)
                if callback: callback(t, x, row)
            return x, history

        # n-D version
        x0 = np.array(obj.init, dtype=float)
        n = x0.size
        simplex = [x0]
        for i in range(n):
            e = np.zeros(n); e[i] = 1.0
            simplex.append(x0 + 0.05 * e)  # small initialization
        simplex = np.array(simplex)

        def f(v): return obj.f(v)

        history: List[Dict[str, Any]] = []
        for t in range(steps):
            vals = np.array([f(v) for v in simplex])
            idx = np.argsort(vals)
            simplex = simplex[idx]; vals = vals[idx]
            x_best, x_worst, x_sec = simplex[0], simplex[-1], simplex[-2]

            x_centroid = simplex[:-1].mean(axis=0)
            x_reflect = x_centroid + self.alpha * (x_centroid - x_worst)
            f_reflect = f(x_reflect)

            if f_reflect < vals[0]:
                x_expand = x_centroid + self.gamma * (x_reflect - x_centroid)
                if f(x_expand) < f_reflect:
                    simplex[-1] = x_expand
                else:
                    simplex[-1] = x_reflect
            elif f_reflect < vals[-2]:
                simplex[-1] = x_reflect
            else:
                if f_reflect < vals[-1]:
                    x_contract = x_centroid + self.rho * (x_reflect - x_centroid)
                else:
                    x_contract = x_centroid + self.rho * (x_worst - x_centroid)
                if f(x_contract) < vals[-1]:
                    simplex[-1] = x_contract
                else:
                    # shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + self.sigma * (simplex[i] - simplex[0])

            row = {"f": float(vals[0]), "grad_norm2": 0.0, "lr": 0.0}
            history.append(row)
            if callback: callback(t, simplex[0], row)

        x_star = simplex[0]
        return (float(x_star) if x_star.size == 1 else x_star), history
