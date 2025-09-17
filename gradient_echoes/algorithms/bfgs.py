import numpy as np

def bfgs(f, grad_f, x0, tol=1e-6, max_iter=100):
    n = len(x0)
    B = np.eye(n)  # Initial Hessian approx
    x = x0
    g = grad_f(x)
    for _ in range(max_iter):
        if np.linalg.norm(g) < tol:
            return x
        p = -np.dot(np.linalg.inv(B), g)  # Direction
        # Line search for step size (implement or use scipy.optimize.line_search)
        alpha = 1.0  # Placeholder
        x += alpha * p
        g_new = grad_f(x)
        s = alpha * p
        y = g_new - g
        B += np.outer(y, y) / np.dot(y, s) - np.outer(np.dot(B, s), np.dot(s, B)) / np.dot(s, np.dot(B, s))
        g = g_new
    return x
#Enhance Repo: Benchmark against exact Newton on quadratic problems; include proof of positive definiteness
