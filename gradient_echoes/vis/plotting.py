# gradient_echoes/vis/plotting.py
from __future__ import annotations
from typing import Dict, List, Any, Optional

def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        raise RuntimeError("Matplotlib is required for plotting. Try: pip install matplotlib") from e

def plot_histories(histories: Dict[str, List[Dict[str, Any]]],
                   y: str = "f",
                   title: Optional[str] = None,
                   savepath: Optional[str] = None):
    """Plot metric 'y' from multiple optimizer histories.
    histories: {"Adam": hist, "SGD": hist, ...}, where hist is list of dicts (f, grad_norm2, lr,...)
    """
    ensure_matplotlib()
    import matplotlib.pyplot as plt

    plt.figure()
    for name, hist in histories.items():
        ys = [row.get(y, None) for row in hist]
        xs = list(range(len(ys)))
        plt.plot(xs, ys, label=name)

    plt.xlabel("step")
    plt.ylabel(y)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=160)
    else:
        plt.show()
