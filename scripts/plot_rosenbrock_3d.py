import numpy as np
import matplotlib.pyplot as plt
from gradient_echoes.core.objective import Objective
from gradient_echoes.classical import Adam
from gradient_echoes.core.callbacks import History
from mpl_toolkits.mplot3d import Axes3D  # noqa

def rosen(v):
    x, y = v
    return (1 - x)**2 + 100*(y - x**2)**2

def rosen_grad(v):
    x, y = v
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])

# problem
obj = Objective(f=rosen, grad=rosen_grad, init=np.array([-1.5, 2.0]))
opt = Adam(lr=0.002)
H = History()
_, _ = opt.minimize(obj, steps=1200, callback=H)

traj = np.array([rec["x"] for rec in H.records if "x" in rec])

# plot
X = np.linspace(-2, 2, 200)
Y = np.linspace(-1, 3, 200)
Xg, Yg = np.meshgrid(X, Y)
Z = (1 - Xg)**2 + 100*(Yg - Xg**2)**2

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xg, Yg, Z, rstride=5, cstride=5, alpha=0.7, cmap="viridis")
ax.plot(traj[:, 0], traj[:, 1], [rosen(p) for p in traj], "r-", lw=2, label="Adam trajectory")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
ax.set_title("Rosenbrock surface with Adam trajectory")
plt.tight_layout()
plt.savefig("docs/assets/rosenbrock_adam_3d.png", dpi=200)
print("Saved docs/assets/rosenbrock_adam_3d.png")
