import numpy as np
from scipy.linalg import solve

def newtons_method(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        if np.linalg.norm(g) < tol:
            return x
        delta = solve(H, -g)  # Solve H * delta = -g
        x += delta
    return x

# Example: f(x) = x^2 + sin(x), with grad and hess functions defined.
#Add a notebook solving Rosenbrock function; compare runtime with conjugate gradient.
