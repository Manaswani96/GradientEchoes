import numpy as np
from gradient_echoes.core import minimize
from gradient_echoes.algorithms import GradientDescent, ParticleSwarm, BFGSWrapper

def rosenbrock(x):
    x = np.asarray(x)
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0])

    print("Running Gradient Descent...")
    gd = GradientDescent(lr=1e-3, max_iters=2000)
    r_gd = minimize(gd, rosenbrock, x0, max_iters=2000)
    print("GD -> best:", r_gd.best_value, "x:", r_gd.x_best)

    print("\nRunning Particle Swarm...")
    pso = ParticleSwarm(n_particles=40, max_iters=300)
    r_pso = minimize(pso, rosenbrock, x0, max_iters=300, seed=42)
    print("PSO -> best:", r_pso.best_value, "x:", r_pso.x_best)

    try:
        print("\nRunning BFGS (scipy)...")
        bfgs = BFGSWrapper()
        r_bfgs = minimize(bfgs, rosenbrock, x0)
        print("BFGS -> best:", r_bfgs.best_value, "x:", r_bfgs.x_best)
    except Exception as e:
        print("BFGS skipped:", e)
