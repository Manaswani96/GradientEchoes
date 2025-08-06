import numpy as np
import matplotlib.pyplot as plt

# Problem Definition
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])
x0 = np.array([0.0, 0.0])

def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x

# Conjugate Gradient Method with print
def conjugate_gradient(A, b, x0, tol=1e-6, max_iter=1000):
    print("\nConjugate Gradient Method")
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    xs = [x.copy()]
    for i in range(max_iter):
        Ap = A @ p
        alpha = (r.T @ p) / (p.T @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        xs.append(x.copy())
        print(f"Iter {i+1:2d}: x = {x}, f(x) = {f(x):.6f}, ||r|| = {np.linalg.norm(r):.6e}")
        if np.linalg.norm(r_new) < tol:
            break
        beta = (r_new.T @ r_new) / (r.T @ r)
        p = r_new + beta * p
        r = r_new
    return np.array(xs)

# Run and print iterations
xs_cg = conjugate_gradient(A, b, x0)

# Grid for contour
x_vals = np.linspace(-2, 4, 400)
y_vals = np.linspace(-6, 4, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([f(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

# Plotting
plt.figure(figsize=(12, 6))
plt.contour(X, Y, Z, levels=40, cmap='viridis', alpha=0.8)


# Plot Conjugate Gradient
for i in range(len(xs_cg)-1):
    x_curr, x_next = xs_cg[i], xs_cg[i+1]
    plt.plot([x_curr[0], x_next[0]], [x_curr[1], x_next[1]], 'bs--', label='Conjugate Gradient' if i==0 else "", lw=2)
    plt.arrow(x_curr[0], x_curr[1],
              x_next[0]-x_curr[0], x_next[1]-x_curr[1],
              head_width=0.1, length_includes_head=True, color='blue', alpha=0.6)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Step-by-step Comparison: Steepest Descent vs. Conjugate Gradient')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
