import numpy as np

def simulated_annealing(f, x0, T0=100, cooling_rate=0.95, min_T=1e-5, max_iter=1000):
    x = x0
    fx = f(x)
    T = T0
    for _ in range(max_iter):
        if T < min_T:
            break
        x_new = x + np.random.uniform(-1, 1, len(x))  # Perturb
        f_new = f(x_new)
        delta = f_new - fx
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            x, fx = x_new, f_new
        T *= cooling_rate
    return x
#Enhance Repo: Apply to TSP (Traveling Salesman); doc Metropolis-Hastings connection.
