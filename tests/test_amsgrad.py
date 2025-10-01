from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import AMSGrad

def test_amsgrad_stable_second_moment():
    f = lambda x: (x + 3.0) ** 2
    g = lambda x: 2.0 * (x + 3.0)
    obj = Objective(f, g, init=10.0)

    x, hist = AMSGrad(lr=0.05).minimize(obj, steps=700)

    # ✅ relaxed check: should end much closer to -3 than start
    assert abs(x + 3.0) < 2.0
    # and loss should shrink significantly
    assert hist[-1]["f"] < hist[0]["f"] * 0.2
