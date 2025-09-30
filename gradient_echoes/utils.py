from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Record:
    step: int
    f: float
    grad_norm2: float
    lr: float

def tabulate(history: List[Dict[str, Any]], every: int = 20) -> str:
    rows = history[-every:]
    if not rows: return ""
    keys = rows[0].keys()
    header = " | ".join(keys)
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(" | ".join(str(round(r[k], 6)) if isinstance(r[k], float) else str(r[k]) for k in keys))
    return "\n".join(lines)

def set_seed(seed: Optional[int]):
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed if seed is not None else 0)
    except Exception:
        pass
