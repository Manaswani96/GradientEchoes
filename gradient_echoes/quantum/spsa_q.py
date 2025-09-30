from __future__ import annotations
from typing import Callable, Any, Optional
from ..classical.spsa import SPSA

class QSPSA:
    """Quantum SPSA convenience wrapper.
    Accepts expectation(theta) and handles vector/scalar params via underlying SPSA.
    """
    def __init__(self, a=0.2, c=0.1, alpha=0.602, gamma=0.101):
        self._spsa = SPSA(a=a, c=c, alpha=alpha, gamma=gamma)

    def minimize(self, expectation: Callable[[Any], float], init_params: Any, steps: int = 300, callback: Optional[callable] = None):
        # emulate Objective with only f and projection identity
        class _Obj:
            def __init__(self, f, init): self.f, self.init = f, init
            def project(self, x): return x
        obj = _Obj(expectation, init_params)
        return self._spsa.minimize(obj, steps=steps, callback=callback)
