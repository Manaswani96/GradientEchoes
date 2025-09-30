# examples/rmsprop_logistic.py
# Logistic regression on a synthetic 2D dataset using RMSProp.
import math
import random

try:
    import numpy as np
except Exception:
    np = None

from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import RMSProp
from gradient_echoes.core.callbacks import LogEvery

random.seed(0)
if np is None:
    raise RuntimeError("This example needs NumPy. pip install numpy")

# --- synthetic data
n, d = 200, 2
X = np.random.randn(n, d)
true_w = np.array([1.5, -2.0])
y = (np.dot(X, true_w) + 0.25 * np.random.randn(n) > 0.0).astype(float)

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

def loss(w):
    z = X @ w
    p = sigmoid(z)
    # average negative log-likelihood with tiny L2
    eps = 1e-8
    return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 1e-3 * (w @ w))

def grad(w):
    z = X @ w
    p = sigmoid(z)
    g = X.T @ (p - y) / len(y) + 2e-3 * w
    return g

w0 = np.zeros(d)
obj = Objective(f=loss, grad=grad, init=w0)

opt = RMSProp(lr=0.05, alpha=0.99, centered=True)
log = LogEvery(20, prefix="[rmsprop-logistic]")
w_star, hist = opt.minimize(obj, steps=400, callback=log)

print("w* =", np.round(w_star, 3), "loss* =", round(hist[-1]["f"], 6))
