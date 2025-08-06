import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg

x = np.array([52, 63, 45, 36, 72, 65, 47, 25])
y = np.array([62, 53, 51, 25, 79, 43, 60, 33])
n = len(x)


X = np.vstack((np.ones(n), x)).T
print(X)
A = X.T @ X  # Shape (2, 2) #q
b_vec = X.T @ y  # Shape (2,) #b
theta = np.zeros(2)  # Initial guess for [a, b]  #x
r = b_vec - A @ theta  # doubt g
p = r  #  direction

# Conjugate Gradient Iteration
max_iter = 1000  # Max number of iterations
tol = 1e-6  # Tolerance for convergence

for k in range(max_iter):
    Ap = A @ p  # Matrix-vector multiplication
    #alpha = r.T @ r / (p.T @ Ap) # this formula is for steepest
    #conjugate:
    alpha=np.dot(r,r)/np.dot(p,Ap)
    theta = theta + alpha * p  # Update the solution vector
    r_new = r - alpha * Ap  # Update residual

    if np.linalg.norm(r_new) < tol:
        break
    beta = r_new.T @ r_new / (r.T @ r)  # Compute new search direction coefficient
    p = r_new + beta * p  # Update search direction
    r = r_new  # Update residual

# Extracting a and b from the solution vector theta
a, b = theta
print(f"Intercept (a): {a:.4f}, Slope (b): {b:.4f}")
# fitted values (ŷ)
y_hat = X @ theta  # Fitted y values

plt.scatter(x, y, label='Scattered data', color='blue')
plt.plot(x, y_hat, label='Fitted line', color='red')
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Least Squares Fit using Conjugate Gradient")
plt.legend()
plt.grid(True)
plt.show()

print("Fitted values (ŷ):")
print(y_hat)


