from __future__ import annotations
from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass, field

class Callback(Protocol):
    def __call__(self, step: int, x, info: Dict[str, Any]) -> None: ...

@dataclass
class History:
    keep: int = 1000
    records: List[Dict[str, Any]] = field(default_factory=list)

    def __call__(self, step: int, x, info: Dict[str, Any]) -> None:
        row = {"step": step, **info}
        self.records.append(row)
        if len(self.records) > self.keep:
            self.records.pop(0)

@dataclass
class LogEvery:
    n: int = 50
    prefix: str = "[echo]"
    def __call__(self, step: int, x, info: Dict[str, Any]) -> None:
        if step % self.n == 0 or info.get("done", False):
            f = info.get("f"); g2 = info.get("grad_norm2")
            print(f"{self.prefix} step={step:05d} f={f:.6f} ||g||^2={g2:.3e}")
