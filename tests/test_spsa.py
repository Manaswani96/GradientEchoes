from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import SPSA

def test_spsa_reaches_minimum_on_noisy_quadratic():
    import random
    random.seed(0)

    def f(x):
        noise = 0.01 * (random.random() - 0.5)
        return (x - 2.0) ** 2 + noise

    obj = Objective(f=f, grad=lambda x: 0.0, init=0.0)
    x, hist = SPSA(a=0.2, c=0.1).minimize(obj, steps=400)
    # It should end closer to 2 than the start (0.0)
    assert abs(x - 2.0) < 2.0
    # And loss should be reduced substantially
    assert hist[-1]["f"] < hist[0]["f"] * 0.5
