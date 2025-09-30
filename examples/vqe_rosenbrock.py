# Not a real quantum circuit—just shaped like VQE to show the interface.
from math import pow
from gradient_echoes.quantum import VQE

def rosenbrock(theta):
    x, y = theta
    return (1 - x)**2 + 100 * (y - x**2)**2

def grad_rosenbrock(theta):
    x, y = theta
    dfdx = -2*(1 - x) - 400*x*(y - x**2)
    dfdy = 200*(y - x**2)
    return (dfdx, dfdy)

theta0 = (0.0, 0.0)
opt = VQE()  # uses Adam under the hood
theta_star, hist = opt.minimize(rosenbrock, grad_rosenbrock, theta0, steps=1000)
print("theta* =", tuple(round(t, 4) for t in theta_star), "f* =", round(hist[-1]["f"], 6))
