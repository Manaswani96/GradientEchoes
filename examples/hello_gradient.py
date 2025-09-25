# examples/hello_gradient.py
import numpy as np
import matplotlib.pyplot as plt
from gradient_echoes.optimizers import GD, Adam

# A tiny 2D Rosenbrock-like demo (but simple to visualize)
def rosenbrock(x):
    # x is 2-d vector
    a = 1.0
    b = 100.0
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    a = 1.0
    b = 100.0
    dx = np.zeros_like(x)
    dx[0] = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
    dx[1] = 2*b*(x[1] - x[0]**2)
    return dx

def run_demo(optim, steps=200):
    x = np.array([-1.5, 1.5])
    traj = [x.copy()]
    for i in range(steps):
        g = grad_rosenbrock(x)
        x = optim.step(x, g)
        traj.append(x.copy())
    return np.array(traj)

def plot_traj(traj, title):
    xs = traj[:,0]; ys = traj[:,1]
    plt.figure(figsize=(6,5))
    plt.plot(xs, ys, marker='o', markersize=3, linewidth=1)
    plt.scatter([1.0],[1.0], color='red', label='min at (1,1)')
    plt.title(title)
    plt.xlabel('x0'); plt.ylabel('x1')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    gd = GD(lr=1e-3)
    adam = Adam(lr=2e-3)

    traj_gd = run_demo(gd, steps=2000)
    traj_adam = run_demo(adam, steps=2000)

    print("GD final:", traj_gd[-1])
    print("Adam final:", traj_adam[-1])

    plot_traj(traj_gd, "Trajectory: GD")
    plot_traj(traj_adam, "Trajectory: Adam")
