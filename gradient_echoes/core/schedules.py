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
    """OneCycle learning rate policy (Smith, 2018) with triangular cosine shape.
    - total_steps: total training steps
    - max_lr: peak LR
    - pct_start: fraction of steps spent increasing to max_lr
    - div_factor: initial_lr = max_lr / div_factor
    - final_div_factor: final_lr = max_lr / final_div_factor
    """
    def __init__(self, total_steps: int, max_lr: float, pct_start: float = 0.3, div_factor: float = 25.0, final_div_factor: float = 1e4):
        self.T = max(1, int(total_steps))
        self.max_lr = float(max_lr)
        self.pct_start = float(pct_start)
        self.div = float(div_factor)
        self.final_div = float(final_div_factor)

        self.T_up = max(1, int(self.T * self.pct_start))
        self.T_down = max(1, self.T - self.T_up)
        self.lr_start = self.max_lr / self.div
        self.lr_end = self.max_lr / self.final_div

    def __call__(self, t: int) -> float:
        t = max(0, min(t, self.T - 1))
        if t < self.T_up:
            # cosine from lr_start -> max_lr
            cos_a = (1 + cos_pi(t / self.T_up)) / 2.0
            return self.max_lr - (self.max_lr - self.lr_start) * cos_a
        else:
            # cosine from max_lr -> lr_end
            td = t - self.T_up
            cos_b = (1 + cos_pi(td / self.T_down)) / 2.0
            return self.lr_end + (self.max_lr - self.lr_end) * cos_b

def cos_pi(x: float) -> float:
    from math import cos, pi
    return cos(pi * x)
