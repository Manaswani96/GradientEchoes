# examples/qspsa_vqe_toy.py
# QSPSA on a toy expectation: E(theta) = sin^2(theta) + small noise.
import math
import random
random.seed(42)

from gradient_echoes.quantum import QSPSA

def expectation(theta):
    # scalar theta; noise simulates sampling error
    base = math.sin(theta)**2
    noise = 0.01 * (random.random() - 0.5)
    return base + noise

theta0 = 1.3
opt = QSPSA(a=0.3, c=0.15)  # fairly gentle
theta_star, hist = opt.minimize(expectation, theta0, steps=300)

print("theta* ≈", round(float(theta_star), 4), "E* ≈", round(hist[-1]["f"], 6))
