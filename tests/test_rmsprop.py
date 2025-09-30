# tests/test_rmsprop.py
from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import RMSProp

def test_rmsprop_converges_on_quadratic():
    f = lambda x: (x - 3.0) ** 2
    g = lambda x: 2.0 * (x - 3.0)
    obj = Objective(f, g, init=0.0)
    x, hist = RMSProp(lr=0.1, centered=True).minimize(obj, steps=300)
    assert abs(x - 3.0) < 1e-2
    assert hist[-1]["f"] < 1e-4
