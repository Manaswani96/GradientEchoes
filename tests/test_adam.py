from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import Adam

def test_adam_converges_on_quadratic():
    f = lambda x: (x + 2.0) ** 2
    g = lambda x: 2.0 * (x + 2.0)
    obj = Objective(f, g, init=10.0)
    x, hist = Adam(lr=0.1).minimize(obj, steps=200)
    assert abs(x + 2.0) < 1e-2
    assert hist[-1]["f"] < 1e-4
