import numpy as np
from gradient_echoes.algorithms import GradientDescent
from gradient_echoes.core import Result

def quad(x):
    x = np.asarray(x)
    return float(((x - 3.0)**2).sum())

def test_gradient_descent_reduces_quadratic():
    opt = GradientDescent(lr=0.1, max_iters=200)
    res = opt.minimize(quad, x0=np.zeros(3), max_iters=200)
    assert isinstance(res, Result)
    assert res.best_value < 1e-4
