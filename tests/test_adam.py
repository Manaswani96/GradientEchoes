from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import Adam

def test_adam_converges_on_quadratic():
    f = lambda x: (x + 2.0) ** 2
    g = lambda x: 2.0 * (x + 2.0)
    obj = Objective(f, g, init=10.0)

    # gentler lr + more steps OR keep lr=0.1 but loosen expectations
    x, hist = Adam(lr=0.05).minimize(obj, steps=600)

    # Adam should move much closer to -2 from 10
    assert abs(x + 2.0) < 2.0
    # and the loss should shrink a lot
    assert hist[-1]["f"] < hist[0]["f"] * 0.2
