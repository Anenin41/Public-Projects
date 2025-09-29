import numpy as np
import matplotlib.pyplot as plt

# Parameters
lambda_reg = 0.1  # Regularization parameter
tau = 0.02        # Primal step size
sigma = 0.02      # Dual step size
num_iter = 100    # Number of iterations

# Generate synthetic blurred image
np.random.seed(42)
X_blur = np.random.rand(256, 256)

# Define the gradient operator
def gradient_operator(X):
    grad_x = np.roll(X, -1, axis=0) - X
    grad_y = np.roll(X, -1, axis=1) - X
    return np.stack((grad_x, grad_y), axis=-1)

# Define the divergence operator (adjoint of gradient)
def divergence_operator(P):
    dx = P[..., 0] - np.roll(P[..., 0], 1, axis=0)
    dy = P[..., 1] - np.roll(P[..., 1], 1, axis=1)
    return dx + dy

# Define the primal-dual update function
def primal_dual_update(X, Y):
    # Primal update
    grad_X = gradient_operator(X)
    X_new = X - tau * (lambda_reg * divergence_operator(grad_X - Y))

    # Dual update using Moreau’s identity
    Y_new = Y + sigma * gradient_operator(2 * X_new - X)
    Y_new = np.sign(Y_new) * np.maximum(np.abs(Y_new) - sigma, 0)  # Soft-thresholding

    return X_new, Y_new

# Initialize variables
X = np.copy(X_blur)
Y = np.zeros((256, 256, 2))

# Run the primal-dual algorithm
for k in range(num_iter):
    X, Y = primal_dual_update(X, Y)
    if k % 10 == 0:
        print(f"Iteration {k}: Residual = {np.linalg.norm(X - X_blur):.4f}")

# Display results
plt.subplot(1, 2, 1)
plt.title("Blurred Image")
plt.imshow(X_blur, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Denoised Image")
plt.imshow(X, cmap='gray')
plt.show()
