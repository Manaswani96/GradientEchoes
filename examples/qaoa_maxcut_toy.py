from math import cos, sin, pi
from itertools import product
import random

from gradient_echoes.quantum import QAOA

# 4-node cycle graph C4 edges
EDGES = [(0,1),(1,2),(2,3),(3,0)]

def classical_cut_value(bitstring):
    # counts edges with different bits
    return sum(1 if bitstring[u] != bitstring[v] else 0 for (u,v) in EDGES)

# Smooth surrogate of QAOA expectation for demo (not exact QAOA physics).
# We build a periodic function of gammas/betas whose minima correlate with high cuts.
def cost_expectation(gammas, betas):
    p = len(gammas)
    random.seed(0)  # deterministic
    val = 0.0
    for (u,v) in EDGES:
        # toy: interference-like term
        s = 0.0
        for l in range(p):
            s += cos(gammas[l]) * sin(betas[l]) + 0.5 * cos(2*gammas[l]) * sin(2*betas[l])
        # encourage edge disagreement
        val += 1.5 - s
    # normalize
    return val / len(EDGES)

p = 2
init_g = [0.3]*p
init_b = [0.2]*p

runner = QAOA(p=p)
theta_star, hist = runner.minimize(cost_expectation, init_g, init_b, steps=300)

print("theta* =", tuple(round(t, 4) for t in theta_star))
print("cost*  =", round(hist[-1]["f"], 6))
