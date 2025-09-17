import numpy as np
from typing import Callable, Optional
from ..core import Result

def _num_grad(func, x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = eps
        g[i] = (func(x + d) - func(x - d)) / (2*eps)
    return g

class GradientDescent:
    """Simple (vanilla) gradient descent with optional numeric gradient.
    Educational, but useful for small smooth problems.
    """
    def __init__(self, lr: float = 1e-2, tol: float = 1e-6, max_iters: int = 1000, verbose: bool = False):
        self.lr = lr
        self.tol = tol
        self.max_iters = max_iters
        self.verbose = verbose

    def minimize(self, func: Callable, x0, *, grad: Optional[Callable]=None, max_iters: Optional[int]=None,
                 callback: Optional[Callable]=None, seed: Optional[int]=None):
        rng = np.random.default_rng(seed)
        x = np.asarray(x0, dtype=float).copy()
        nit = 0
        history = []
        max_iters = max_iters or self.max_iters

        fx = float(func(x))
        history.append((0, fx))
        if self.verbose:
            print(f"[GD] iter 0, value={fx:.6g}")

        while nit < max_iters:
            nit += 1
            g = grad(x) if (grad is not None) else _num_grad(lambda xx: func(xx), x)
            x = x - self.lr * g
            fx = float(func(x))
            history.append((nit, fx))
            if callback:
                callback(nit, x, fx)
            if self.verbose and (nit % 50 == 0):
                print(f"[GD] iter {nit}, value={fx:.6g}")
            if np.linalg.norm(g) * self.lr < self.tol:
                break

        return Result(x_best=x, best_value=fx, nit=nit, history=history)
