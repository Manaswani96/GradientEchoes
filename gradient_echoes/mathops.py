from __future__ import annotations
from typing import Any, Iterable

try:
    import numpy as np
except Exception:  # numpy is optional
    np = None  # type: ignore

def is_array(x: Any) -> bool:
    return np is not None and isinstance(x, np.ndarray)

def zeros_like(x: Any):
    if is_array(x): return np.zeros_like(x)
    return 0.0

def add(x, y):
    if is_array(x) or is_array(y): return x + y
    return float(x) + float(y)

def sub(x, y):
    if is_array(x) or is_array(y): return x - y
    return float(x) - float(y)

def mul(a, x):
    if is_array(x): return a * x
    return float(a) * float(x)

def norm2(x) -> float:
    if is_array(x): return float((x * x).sum())
    return float(x) * float(x)
