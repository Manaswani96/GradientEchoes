import numpy as np
from typing import Callable, Optional
from ..core import Result

class ParticleSwarm:
    """Simple PSO for continuous optimization.
    Not heavily optimized — educational + useful for multi-modal functions.
    """
    def __init__(self, n_particles=30, inertia=0.7, cognitive=1.5, social=1.5, max_iters=200, verbose=False):
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iters = max_iters
        self.verbose = verbose

    def minimize(self, func: Callable, x0, *, grad: Optional[Callable]=None, max_iters: Optional[int]=None,
                 callback: Optional[Callable]=None, seed: Optional[int]=None):
        rng = np.random.default_rng(seed)
        x0 = np.asarray(x0, dtype=float)
        dim = x0.size
        max_iters = max_iters or self.max_iters

        # Initialize swarm around x0
        pos = x0 + rng.normal(scale=1.0, size=(self.n_particles, dim))
        vel = rng.normal(scale=0.1, size=(self.n_particles, dim))
        pbest = pos.copy()
        pbest_val = np.array([func(p) for p in pos])
        gbest_idx = int(np.argmin(pbest_val))
        gbest = pbest[gbest_idx].copy()
        gbest_val = float(pbest_val[gbest_idx])
        history = [(0, gbest_val)]

        for t in range(1, max_iters+1):
            r1 = rng.random((self.n_particles, dim))
            r2 = rng.random((self.n_particles, dim))
            vel = (self.inertia * vel
                   + self.cognitive * r1 * (pbest - pos)
                   + self.social * r2 * (gbest - pos))
            pos = pos + vel

            # evaluate
            vals = np.array([func(p) for p in pos])
            improved = vals < pbest_val
            pbest[improved] = pos[improved]
            pbest_val[improved] = vals[improved]

            local_best_idx = int(np.argmin(pbest_val))
            if pbest_val[local_best_idx] < gbest_val:
                gbest_val = float(pbest_val[local_best_idx])
                gbest = pbest[local_best_idx].copy()

            history.append((t, gbest_val))
            if callback:
                callback(t, gbest.copy(), gbest_val)
            if self.verbose and (t % 50 == 0):
                print(f"[PSO] iter {t}, best={gbest_val:.6g}")

        return Result(x_best=gbest, best_value=gbest_val, nit=t, history=history)
