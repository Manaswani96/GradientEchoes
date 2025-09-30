from __future__ import annotations
from typing import Callable, Any, Optional, List, Dict
from ..core.objective import Objective
from ..core.oracle import Oracle
from ..core.callbacks import Callback
from ..classical.adam import Adam

class VQE:
    """Tiny VQE loop facade.
    You provide: expectation(theta) -> float and gradient(theta) -> like-theta.
    We wrap it in Objective and optimize with a chosen classical optimizer (Adam default).
    """
    def __init__(self, optimizer=None):
        self.optimizer = optimizer or Adam(lr=0.05)

    def minimize(self, expectation: Callable[[Any], float], grad: Callable[[Any], Any], init_params: Any, steps: int = 200, callback: Optional[Callback] = None):
        obj = Objective(f=expectation, grad=grad, init=init_params)
        x, hist = self.optimizer.minimize(obj=obj, oracle=Oracle(obj.f, obj.grad), steps=steps, callback=callback)
        return x, hist
