# simple optimizers for learning + demos

from dataclasses import dataclass
import numpy as np

@dataclass
class GD:
    lr: float = 0.1

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Simple gradient descent update"""
        return x - self.lr * grad


@dataclass
class Adam:
    lr: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    m: np.ndarray | None = None
    v: np.ndarray | None = None
    t: int = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
