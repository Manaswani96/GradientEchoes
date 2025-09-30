from __future__ import annotations
from typing import Callable, Any, Optional, Tuple, List, Dict
from math import pi
from ..core.objective import Objective, Constraint
from ..core.oracle import Oracle
from ..classical.adam import Adam

class QAOA:
    """QAOA loop facade (backend-agnostic).
    You provide cost_expectation(gammas, betas) -> float.
    We optimize concatenated params [(gamma_1,...,gamma_p),(beta_1,...,beta_p)] using Adam.
    """
    def __init__(self, p: int, optimizer=None, beta_bounds: Tuple[float, float]=(0.0, pi/2), gamma_bounds: Tuple[float, float]=(0.0, pi)):
        self.p = int(p)
        self.optimizer = optimizer or Adam(lr=0.05)
        self.beta_bounds = beta_bounds
        self.gamma_bounds = gamma_bounds

    def _pack(self, gammas, betas):
        return tuple(gammas) + tuple(betas)

    def _unpack(self, theta):
        return theta[:self.p], theta[self.p:]

    def _box_project(self, theta):
        g_lo, g_hi = self.gamma_bounds
        b_lo, b_hi = self.beta_bounds
        th = list(theta)
        for i in range(self.p):
            th[i] = min(max(th[i], g_lo), g_hi)
        for i in range(self.p, 2*self.p):
            th[i] = min(max(th[i], b_lo), b_hi)
        return tuple(th)

    def minimize(self, cost_expectation: Callable[[Any, Any], float], init_gammas, init_betas, steps: int = 400, callback=None):
        def f(theta):
            g, b = self._unpack(theta)
            return float(cost_expectation(g, b))
        # use finite differences via small helper gradient (cheap, p is small)
        def grad(theta, h: float = 1e-3):
            theta = list(theta)
            g = []
            for i in range(len(theta)):
                t1 = theta.copy(); t2 = theta.copy()
                t1[i] += h; t2[i] -= h
                g.append((f(t1) - f(t2)) / (2*h))
            return tuple(g)

        constraint = Constraint(name="box", project=self._box_project)
        theta0 = self._pack(init_gammas, init_betas)
        obj = Objective(f=f, grad=grad, init=theta0, constraint=constraint)
        x, hist = self.optimizer.minimize(obj, oracle=Oracle(obj.f, obj.grad), steps=steps, callback=callback)
        return x, hist
