# Python Code for the 2nd Computational Exercise of Iterative Algorithms #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 20 Jan 2025 @ 17:58:09 +0100
# Modified: Fri 24 Jan 2025 @ 20:06:58 +0100

# Packages
from RuG_IntroToOptimization import *
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator

# Image Loader
def image_loader():
    X_ref = load_image("test_image.jpg")
    return X_ref

# Generate an Image Plotter with the desired canvas (row & column formation)
def gen_canvas_grid(rows, columns):
    rows = int(rows)
    columns = int(columns)
    canvas = ImagePlotter(rows, columns)
    return canvas

# Apply noise to the image array
def apply_blur(image):
    X_blur = R(image)
    X_blur_transpose = R_T(image)
    return X_blur, X_blur_transpose

# Define prox() operator
def prox(z, sigma):
    return np.sign(z) * np.maximum(np.abs(z) - (1 / sigma), 0)

# Define Primal-Dual Algorithm
def primal_dual(image, lambda_par, tau, sigma, tol):
    """
    image: multidimensional numpy array.
    lambda_par: regularization parameter.
    tau: primal-dual algorithm parameter.
    sigma: primal-dual algorithm parameter.
    tol: maximum number of iterations.
    """

    # Initialize Image Blurring
    x_blur, x_blur_T = apply_blur(image)
    
    # Initialize Primal-Dual algorithm parameters
    x = np.copy(image)              # primal variable
    y = np.zeros((256, 256))        # dual variable

    # Main Iteration Loop
    for k in range(tol):
        # Compute x_{k+1}
        x_new = x - tau * (lambda_par * grad_T(grad(x - x_blur) - y))
        
        # Compute y_{k+1}
        y_new = y + sigma * grad(2 * x_new - x) - sigma * prox(y, sigma)
        
        # Update the iteration rules
        x = x_new
        y = y_new

        # Convergence Check
        if k % 10 == 0:
            print(f"Iteration {k}, Residual: {np.linalg.norm(x - x_blur):.4f}")
    return x, y

def admm(blurred_image, lambda_par, gamma, tol=1000):
    """
    blurred_image: Blurred input image.
    R: Operator (function) that applies blurring.
    R_T: Operator (function) that applies the transpose of blurring.
    tol: Maximum number of iterations.
    """

    # Flatten the image to 1D for processing
    b = blurred_image.flatten()
    image_shape = blurred_image.shape
    y = np.zeros((256, 256))
    u = np.zeros_like(y)

    # Define the linear operator that simulates the system Ax = b
    def R_func(x):
        x = x.reshape(image_shape)
        return R(x).flatten()

    def R_transpose_func(x):
        x = x.reshape(image_shape)
        return R_T(x).flatten()

    # Create a LinearOperator of R
    OPERATOR = LinearOperator((b.size, b.size), matvec = R_func, rmatvec = R_transpose_func)
    
    # Solve Ax = b using conjugate gradient
    x, _ = cg(OPERATOR, b, maxiter = tol)

    # Reshape result back to 2D Image
    return x.reshape(image_shape)

# Main
def main():
    print("Iterative Algorithms - Computational Exercise 2")
    print("===============================================")
    print("1. Primal-Dual Algorithm")
    print("2. ADMM Algorith")
    print("Choose desired method (1, 2).")
    method = input("Method of choice: ")
    X_ref = image_loader()
    X_blur, X_blur_transpose = apply_blur(X_ref)
    if method == "1":
        X_prime, Y_dual = primal_dual(X_ref, 0.2, 0.01, 0.2, 500)
        canvas = gen_canvas_grid(1, 4)
        canvas.plot_image(X_ref, "Original Image", 0, 0)
        canvas.plot_image(X_blur, "Blurry Image", 0, 1)
        canvas.plot_image(X_
        canvas.plot_image(X_prime, "Prime", 0, 1)
        canvas.show()
    elif method == "2":
        X_prime = admm(X_blur, 1.0, 0.05, 500)
        canvas = gen_canvas_grid(1, 2)
        canvas.plot_image(X_ref, "Original Image", 0, 0)
        canvas.plot_image(X_prime, "Prime", 0, 1)
        canvas.show()
    else:
        print(f"{method} is not a valid choice.")
    print("Terminating.")

main()
