import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from scipy.sparse.linalg import cg, LinearOperator
from RuG_IntroToOptimization import *

# Parameters
lambda_reg = 1  # Regularization parameter
gamma = 0.3       # ADMM penalty parameter
num_iter = 11    # Number of iterations

# Load or create blurred image
np.random.seed(42)
X = load_image("test_image.jpg")
X_blur = R(X)
X_blur_T = R_T(X)
#X_blur = np.random.rand(256, 256)

# Soft-thresholding function for L1 norm
def soft_thresholding(V, threshold):
    return np.sign(V) * np.maximum(np.abs(V) - threshold, 0)

# Initialize variables
X = np.copy(X_blur)
Y = np.zeros((2, 256, 256))
U = np.zeros((2, 256, 256))
c = np.zeros((2, 256, 256))  # Constraint term

# Define linear operator function for Conjugate Gradient solver
def linear_operator(X_flat):
    X = X_flat.reshape(X_blur.shape)
    return (lambda_reg * X + gamma * grad_T(grad(X))).ravel()

# ADMM iterations
for k in range(num_iter):
    # Compute right-hand side
    V = -grad(X) + (1 / gamma) * U
    Y = soft_thresholding(V, 1 / gamma)

    grad_X = grad(X)
    rhs = (lambda_reg * X_blur_T.flatten() + gamma * grad_T(Y - (1 / gamma) * U).flatten())

    # Solve X update using Conjugate Gradient method
    X_new, _ = cg(LinearOperator((X.size, X.size), matvec=linear_operator), rhs, x0=X.flatten(), maxiter=100)
    X_new = X_new.reshape(X.shape)

    # Update U (dual variable update)
    U += gamma * (grad(X_new) - Y)

    # Convergence check
    if k % 10 == 0:
        residual = np.linalg.norm(grad(X_new) - Y)
        print(f"Iteration {k}: Residual = {residual:.4f}")

    X = X_new

# Display results
plt.subplot(1, 2, 1)
plt.title("Blurred Image")
plt.imshow(X_blur, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Denoised Image (ADMM)")
plt.imshow(X, cmap='gray')
plt.show()

