from __future__ import annotations
from typing import Callable, Protocol, Any, Dict, Tuple, Iterable, Optional, Union

Scalar = float
Vector = Union[float, Any]  # Any: numpy.ndarray or torch.Tensor later
PRNGKey = Optional[int]

class StepFn(Protocol):
    def __call__(self, t: int) -> float: ...
