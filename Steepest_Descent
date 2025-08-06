import numpy as np
import matplotlib.pyplot as plt

# Problem Definition
A = np.array([[3, 2], [2, 6]])
b = np.array([2, -8])
x0 = np.array([0.0, 0.0])

def f(x):
    return 0.5 * x.T @ A @ x - b.T @ x

# Steepest Descent Method with print
def steepest_descent(A, b, x0, tol=1e-6, max_iter=1000):
    print("\nSteepest Descent Method")
    x = x0.copy()
    xs = [x.copy()]
    for i in range(max_iter):
        r = b - A @ x
        alpha = r.T @ r / (r.T @ A @ r)
        x = x + alpha * r
        xs.append(x.copy())
        print(f"Iter {i+1:2d}: x = {x}, f(x) = {f(x):.6f}, ||r|| = {np.linalg.norm(r):.6e}")
        if np.linalg.norm(r) < tol:
            break
    return np.array(xs)



# Run and print iterations
xs_sd = steepest_descent(A, b, x0)


# Grid for contour
x_vals = np.linspace(-2, 4, 400)
y_vals = np.linspace(-6, 4, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([f(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

# Plotting
plt.figure(figsize=(12, 6))
plt.contour(X, Y, Z, levels=40, cmap='viridis', alpha=0.8)

# Plot Steepest Descent
for i in range(len(xs_sd)-1):
    x_curr, x_next = xs_sd[i], xs_sd[i+1]
    plt.plot([x_curr[0], x_next[0]], [x_curr[1], x_next[1]], 'ro-', label='Steepest Descent' if i==0 else "", lw=2)
    plt.arrow(x_curr[0], x_curr[1],
              x_next[0]-x_curr[0], x_next[1]-x_curr[1],
              head_width=0.1, length_includes_head=True, color='red', alpha=0.6)


plt.xlabel('x')
plt.ylabel('y')
plt.title('Steepest Descent vs. Conjugate Gradient')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
