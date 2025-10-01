from __future__ import annotations
from math import cos, pi

class Constant:
    def __init__(self, lr: float): self.lr = float(lr)
    def __call__(self, t: int) -> float: return self.lr

class StepDecay:
    def __init__(self, lr: float, drop: float = 0.5, every: int = 100):
        self.lr, self.drop, self.every = lr, drop, every
    def __call__(self, t: int) -> float:
        k = t // self.every
        return self.lr * (self.drop ** k)

class CosineDecay:
    def __init__(self, lr: float, T: int):
        self.lr, self.T = lr, max(1, T)
    def __call__(self, t: int) -> float:
        return 0.5 * self.lr * (1 + cos(pi * min(t, self.T) / self.T))

class Warmup:
    def __init__(self, base, warm: int = 50, factor: float = 0.1):
        self.base, self.warm, self.factor = base, warm, factor
    def __call__(self, t: int) -> float:
        if t < self.warm:
            return self.factor * (t + 1) / self.warm * self.base(0)
        return self.base(t - self.warm)
class OneCycle:
    """OneCycle learning rate policy (Smith, 2018) with triangular cosine shape."""
    def __init__(self, total_steps: int, max_lr: float, pct_start: float = 0.3,
                 div_factor: float = 25.0, final_div_factor: float = 1e4):
        self.T = max(1, int(total_steps))
        self.max_lr = float(max_lr)
        self.pct_start = float(pct_start)
        self.div = float(div_factor)
        self.final_div = float(final_div_factor)

        self.T_up = max(1, int(round(self.T * self.pct_start)))
        self.T_down = max(1, self.T - self.T_up)

        self.lr_start = self.max_lr / self.div
        self.lr_end = self.max_lr / self.final_div

    def __call__(self, t: int) -> float:
        # clamp t so we always return defined endpoints
        t = max(0, min(t, self.T - 1))
        if t < self.T_up:
            # cosine from lr_start -> max_lr over T_up steps, hitting max exactly at t=T_up-1
            return _cos_anneal(self.lr_start, self.max_lr, t, self.T_up)
        else:
            td = t - self.T_up
            # cosine from max_lr -> lr_end over T_down steps, hitting lr_end at t=T-1
            return _cos_anneal(self.max_lr, self.lr_end, td, self.T_down)

def _cos_anneal(start: float, end: float, i: int, total: int) -> float:
    """Cosine interpolation that hits exactly start at i=0 and end at i=total-1."""
    if total <= 1:  # degenerate
        return float(end)
    from math import cos, pi
    # map i in [0, total-1] to phase in [0, pi]
    phase = pi * (i / (total - 1))
    w = 0.5 * (1 + cos(phase))  # 1 -> 0 smoothly
    return end + (start - end) * w
