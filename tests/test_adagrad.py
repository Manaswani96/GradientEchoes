from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import AdaGrad

def test_adagrad_converges_and_adapts_step():
    f = lambda x: (x - 7.0) ** 2
    g = lambda x: 2.0 * (x - 7.0)
    obj = Objective(f, g, init=0.0)
    x, hist = AdaGrad(lr=0.8).minimize(obj, steps=300)
    assert abs(x - 7.0) < 1e-2
    assert hist[-1]["f"] < 1e-4
