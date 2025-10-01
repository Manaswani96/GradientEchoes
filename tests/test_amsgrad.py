from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import AMSGrad

def test_amsgrad_stable_second_moment():
    f = lambda x: (x + 3.0) ** 2
    g = lambda x: 2.0 * (x + 3.0)
    obj = Objective(f, g, init=10.0)
    x, hist = AMSGrad(lr=0.1).minimize(obj, steps=300)
    assert abs(x + 3.0) < 1e-2
    assert hist[-1]["f"] < 1e-4
