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
