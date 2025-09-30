from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Optional

# --- geometric hints (kept abstract, usable by QNG/constraints) ----------------
@dataclass
class Manifold:
    name: str = "R^n"
    # metric callback: g(x, v) -> v^T G(x) v (optional)
    metric: Optional[Callable[[Any, Any], float]] = None

@dataclass
class Constraint:
    """Generic constraint; project(x) is a projection operator onto the feasible set."""
    name: str = "unconstrained"
    project: Optional[Callable[[Any], Any]] = None

# --- objective -----------------------------------------------------------------
@dataclass
class Objective:
    """Objective wraps f and grad; intentionally minimal and explicit.

    f: R^n -> R
    grad: R^n -> R^n
    """
    f: Callable[[Any], float]
    grad: Callable[[Any], Any]
    init: Any
    manifold: Manifold = field(default_factory=Manifold)
    constraint: Constraint = field(default_factory=Constraint)
    meta: Dict[str, Any] = field(default_factory=dict)

    def project(self, x):
        return self.constraint.project(x) if self.constraint.project else x
