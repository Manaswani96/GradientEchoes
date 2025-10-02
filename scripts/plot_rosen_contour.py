# scripts/plot_rosen_contour.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import Adam

def rosen(v):
    x, y = v
    return (1 - x)**2 + 100*(y - x**2)**2

def rosen_grad(v):
    x, y = v
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# reproduceable trajectory using Adam-like steps (you can swap to VQE/other)
def adam2d(init, lr=0.002, steps=800):
    x = np.array(init, dtype=float)
    m = np.zeros(2); v = np.zeros(2)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    traj = [x.copy()]
    for t in range(1, steps+1):
        g = rosen_grad(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*(g*g)
        mhat = m / (1 - b1**t)
        vhat = v / (1 - b2**t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        if t % 4 == 0:
            traj.append(x.copy())
    return np.array(traj)

traj = adam2d([-1.5, 2.0], steps=800)

x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2

fig, ax = plt.subplots(figsize=(8,6))
levels = np.logspace(-0.5, 3.5, 25)
cont = ax.contour(X, Y, Z, levels=levels, norm=colors.LogNorm(), cmap="viridis")
ax.plot(traj[:,0], traj[:,1], color="red", linewidth=2, marker="o", markersize=3, markevery=10)
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Rosenbrock contours with Adam trajectory (downsampled)")
ax.clabel(cont, inline=1, fontsize=8, fmt="%.0f")
plt.tight_layout()
plt.savefig("docs/assets/rosenbrock_contour_traj.png", dpi=200)
print("Saved docs/assets/rosenbrock_contour_traj.png")
