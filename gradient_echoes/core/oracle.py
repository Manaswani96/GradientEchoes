from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Optional
from ..mathops import norm2

@dataclass
class Oracle:
    """Deterministic oracle: returns (value, gradient)."""
    value: Callable[[Any], float]
    grad: Callable[[Any], Any]

    def __call__(self, x: Any):
        f = self.value(x)
        g = self.grad(x)
        return f, g, {"grad_norm2": norm2(g)}

@dataclass
class StochasticOracle:
    """Stochastic oracle with optional mini-batching, controlled by 'key'."""
    value: Callable[[Any, Optional[int]], float]
    grad: Callable[[Any, Optional[int]], Any]

    def __call__(self, x: Any, key: Optional[int] = None):
        f = self.value(x, key)
        g = self.grad(x, key)
        return f, g, {"grad_norm2": norm2(g)}
