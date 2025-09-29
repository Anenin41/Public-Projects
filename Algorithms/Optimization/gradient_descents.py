# Python Code that Solve the 1st Computational Exercise of Iterative Algorithms #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Fri 20 Dec 2024 @ 13:03:47 +0100
# Modified: Mon 29 Sep 2025 @ 21:55:02 +0200

# Packages
import numpy as np
import matplotlib.pyplot as plt
import time

# Least Squares Problem (written in the P-version)
def least_squares_problem(x, P, c, d):
    # Equivalent expression for the least squares problem, drawn from the
    # lecture notes, and written using Numpy notation (array optimization).
    equation = 0.5 * x.T @ P @ x + c.T @ x + d
    return equation

# Compute the gradient of the quadratic form
def gradient(x, P, c):
    # Expression for the gradient of the quadratic form.
    equation = P @ x + c
    return equation

# Pre-compute important matrix operations, to reduce the time complexity of the
# different gradient descent implementations.
def precompute(A, b):
    # Matrix P, symmetric and positive semi-definite
    P = A.T @ A
    c = - A.T @ b
    d = 0.5 * np.linalg.norm(b)**2
    return P, c, d

# Compute the L-smoothness constant of the quadratic form, i.e., the largest
# eigenvalue of matrix P. For this, use the Numpy pagkage (linear algebra
# optimization package).
def compute_L_constant(P):
    L = np.max(np.linalg.eigvalsh(P))
    return L

# Matrix and vector generation according to the instructions of the computational
# assignment.
def generator(k):
    # k = matrix size, type: int(), k = 1, ..., 10

    np.random.seed(42)              # Constant seed for use in experiment simulations
    size = 2**k                     # Set the size of the experiment
    A = np.random.randn(size, size) # Generate matrix A
    b = np.random.randn(size)       # Generate vector b
    x = np.random.randn(size)       # Generate vector x
    return A, b, x

# Initialize the least squares problem, by calling in the functions that generate
# the different values and equations of the problem.
def initialize_problem(k):
    # Generate problem parameters (matrix A & vectors b, x)
    A, b, x = generator(k)
    
    # Pre-compute expensive terms in the least squares problem
    P, c, d = precompute(A, b)

    # Compute the value of the quadratic form at (given) x
    Q = least_squares_problem(x, P, c, d)

    # Compute the gradient of the quadratic form at (given) x
    gradient_Q = gradient(x, P, c)
    
    # Generate Output
    return Q, gradient_Q

# Implement the gradient descent algorithm with constant step sizes.
def constant_gradient_descent(A, b, x_0, step, tolerance, max_iterations):
    # Parameters
    # A                 : matrix A of the least squares problem
    # b                 : vector b of the least squares problem
    # x_0               : initial guess of x, mandatory to start the gradient
    #                     descent iteration
    # step              : gradient descent step size, i.e., \gamma
    # tolerance         : acceptable accuracy for the approximate solution, also
    #                     a stopping criterion
    # max_iterations    : maximum budget of computational time, also a stopping
    #                     critetion

    # Transform the problem to its equivalent form using matrix P.
    P, c, d = precompute(A, b)

    # Initialize values before the iteration loop.
    gradient_norms = []              # array that stores the gradient norm values
    iterations = 0                   # initialize the iterations counter
    x = x_0                          # initialize x (as x_0)
    grad = gradient(x, P, c)         # compute initial gradient value
    grad_norm = np.linalg.norm(grad) # compute norm of initial gradient

    # Main iteration loop.
    while iterations < max_iterations:
        if grad_norm <= tolerance:
            break
        else:
            gradient_norms.append(grad_norm)    # store norm in results
            x -= step * grad                    # compute x_{k+1}
            grad = gradient(x, P, c)            # update gradient for next iteration
            grad_norm = np.linalg.norm(grad)    # compute norm of the updated grad
            iterations += 1                     # update iterations counter

    return x, gradient_norms, iterations

# Implement the gradient descent algorithm with vanishing step sizes.
def vanishing_gradient_descent(A, b, x_0, step, p, tolerance, max_iterations):
    # Parameters
    # A                 : matrix A of the least squares problem
    # b                 : vector b of the least squares problem
    # x_0               : initial guess of x, mandatory to start the gradient
    #                     descent algorithm
    # step              : gradient descent step size, i.e., \gamma
    # p                 : power of \gamma (vanishing step)
    # tolerance         : acceptable accuracy for the approximate solution, also
    #                     a stopping criterion
    # max_iterations    : maximum budget of computational time, also a stopping
    #                     criterion

    # Transform the problem to its equivalent form using matrix P.
    P, c, d = precompute(A, b)

    # Initialize values before the iteration loop.
    gradient_norms = []              # array that stores the gradient norm values
    iterations = 0                   # initialize the iterations couunter
    x = x_0                          # initialize x (as x_0)
    grad = gradient(x, P, c)         # compute initial gradient value
    grad_norm = np.linalg.norm(grad) # compute norm of initial gradient

    # Main iteration loop.
    while iterations < max_iterations:
        if grad_norm <= tolerance:
            break
        else:
            gradient_norms.append(grad_norm)    # store norm in results
            step_size = step / ((iterations + 1) ** p)  # vanishing steps
            x -= step_size * grad               # compute x_{k+1}
            grad = gradient(x, P, c)            # update gradient for next iteration
            grad_norm = np.linalg.norm(grad)    # compute norm of the updated grad
            iterations += 1

    return x, gradient_norms, iterations

# Implement the gradient descent algorithm with exact step sizes.
def exact_gradient_descent(A, b, x_0, tolerance, max_iterations):
    # Parameters
    # A                 : matrix A of the least squares problem
    # b                 : vector b of the least squares problem
    # x_0               : initial guess of x, mandatory to start the gradient
    #                     descent algorithm
    # tolerance         : acceptable accuracy for the approximate solution, also
    #                     a stopping criterion
    # max_iterations    : maximum budget of computational time, also a stopping
    #                     criterion

    # Transform the problem to its equivalent form using matrix P.
    P, c, d = precompute(A, b)

    # Initialize values before the iteration loop.
    gradient_norms = []              # array that stores the gradient norm values
    iterations = 0                   # initialize iterations counter
    x = x_0                          # initialize x (as x_0)
    grad = gradient(x, P, c)         # compute initial gradient value
    grad_norm = np.linalg.norm(grad) # compute norm of initial gradient

    # Main iteration loop.
    while iterations < max_iterations:
        if grad_norm <= tolerance:
            break
        else:
            gradient_norms.append(grad_norm)                # store norm in results
            step_size = (grad @ grad) / (grad @ (P @ grad)) # optimal \gamma
            x -= step_size * grad                           # compute x_{k+1}
            grad = gradient(x, P, c)                        # update gradient
            grad_norm = np.linalg.norm(grad)                # update norm
            iterations += 1                                 # update counter

    return x, gradient_norms, iterations

# Implement the gradient descent algorithm with Armijo step sizes.
def armijo_gradient_descent(A, b, x_0, alpha, beta, tolerance, max_iterations):
    # Parameters
    # A                 : matrix A of the least squares problem
    # b                 : vector b of the least squares problem
    # x_0               : initial guess of x, mandatory to start the gradient
    #                     descent algorithm
    # alpha             : slope parameter for Armijo step
    # beta              : reduction factor for Armijo parameter
    # tolerance         : acceptable accuracy for the approximate solution, also
    #                     a stopping criterion
    # max_iterations    : maximum budget of computational time, also a stopping
    #                     criterion

    # Transform the problem to its equivalent form using matrix P.
    P, c, d = precompute(A, b)

    # Initialize values before iteration loop.
    gradient_norms = []              # array that stores the gradient norm values
    iterations = 0                   # initialize iterations counter
    x = x_0                          # initialize x (as x_0)
    grad = gradient(x, P, c)         # compute initial gradient value
    grad_norm = np.linalg.norm(grad) # compute norm of initial gradient

    # Main iteration loop.
    while iterations < max_iterations:
        if grad_norm <= tolerance:
            break
        else:
            gradient_norms.append(grad_norm)    # store norm in results
            step_size = 1.0                     # initial step size arbitrary value
        
            # Nested while loop to find optimal step size that satisfies the Armijo
            # condition (\leq inequality).
            while least_squares_problem(x - step_size * grad, P, c, d) > least_squares_problem(x, P, c, d) - alpha * step_size * np.linalg.norm(grad)**2:
                step_size *= beta        # reduce step according to the beta parameter
            
            x -= step_size * grad               # compute x_{k+1}
            grad = gradient(x, P, c)            # update gradient
            grad_norm = np.linalg.norm(grad)    # update norm
            iterations += 1                     # update iterations counter

    return x, gradient_norms, iterations

# Implement the conjugate gradient algorithm.
def conjugate_gradient(A, b, x_0, tolerance, max_iterations):
    # Parameters
    # A                 : matrix A of the least squares problem
    # b                 : vector b of the least squares problem
    # x_0               : initial guess of x, mandatory to start the gradient
    #                     descent algorithm
    # tolerance         : acceptable accuracy for the approximate solution, also
    #                     a stopping criterion
    # max_iterations    : maximum budget of computational time, also a stopping
    
    # Transform the problem to its equivalent form using matrix P.
    P, c, d = precompute(A, b)

    # Initialize values before iteration loop.
    gradient_norms = []              # array that stores the gradient norm values
    iterations = 0                   # initialize iterations counter
    x = x_0                          # initialize x (as x_0)
    grad = gradient(x, P, c)         # compute initial gradient value
    grad_norm = np.linalg.norm(grad) # compute norm of initial gradient
    d = -grad                        # initial direction

    while grad_norm > tolerance and iterations < max_iterations:
        # Compude P * d_k (common term that shows up frequently)
        Pd = P @ d

        # Compute optimal step size gamma_k
        gamma = - (d.T @ grad) / (d.T @ Pd)

        # Compute x_{k+1}
        x += gamma * d

        # Compute g_{k+1}, g_{k+1} norm and append to result list
        grad_new = grad + gamma * Pd
        grad_norm = np.linalg.norm(grad_new)
        gradient_norms.append(grad_norm)

        # Tolerance condition check
        if grad_norm < tolerance:
            break

        # Compute beta_{k}
        beta = (grad_new.T @ grad_new) / (grad.T @ grad)

        # Update iteration rules
        d = -grad_new + beta * d
        grad = grad_new
        iterations += 1

    return x, gradient_norms, iterations

# Plotting function that generates the desired figures for each method (changes
# to the method are implemented manually each time to take into account the fact
# that not all methods convergent for the same k). 
def plotter(k_values, avg_iterations, method_name):
    plt.figure(figsize=(12, 8))
    plt.plot(k_values, avg_iterations, marker='o', linestyle='-', label=method_name)
    plt.xticks(k_values, labels=[f"{k}" for k in k_values])
    plt.yscale('log')
    plt.xlabel('Matrix Size (2^k)')
    plt.ylabel('Average Number of Iterations')
    plt.title(f"Convergence of {method_name} on Log-Log Scale")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# The following function tests the different gradient descent algorithms
# to check if they work properly. Manually change the parameters inside the 
# function to test each feature, and uncomment the desired method to test it.
def system_testing():
    # Parameters
    # k         : matrix size
    # A, b      : random matrix and vector (seed = 42) of size 2^k x 2^k
    # x_0       : initial guess to jumpstart the algorithms
    # P, c, d   : precompure second version of the Quadratic Form
    # L         : L-smoothness constant
    # step      : optimal step for constant gradient descent
    # max_iter  : maximum up-time
    # tol       : desired accuracy
    # p         : vanishing step denominator power
    # alpha     : Armijo slope parameter 
    # beta      : armijo reduction parameter 

    # Generate least square problem matrices and vectors, and transform them
    # in the equivalent form defined by the P matrix.
    k = 10
    A, b, x = generator(k)
    P, c, d = precompute(A, b)
    x_0 = np.zeros(2**k)
    L = compute_L_constant(P)
    step = 1 / L
    max_iter = 1e+5
    tol = 1e-4
    p = 0.5
    alpha = 0.5
    beta = 0.5
    #x_final, grad_norms, iterations = constant_gradient_descent(A, b, x_0, step, tol, max_iter)
    #x_final, grad_norms, iterations = vanishing_gradient_descent(A, b, x_0, step, p, tol, max_iter)
    #x_final, grad_norms, iterations = exact_gradient_descent(A, b, x_0, tol, max_iter)
    #x_final, grad_norms, iterations = armijo_gradient_descent(A, b, x_0, alpha, beta, tol, max_iter)
    #x_final, grad_norms, iterations = conjugate_gradient(A, b, x_0, tol, max_iter)

    # Output
    print("Tolerance:", tol)
    print("Allowed Budget:", max_iter)
    print("L-smoothness constant:", L)
    print("Optimal constant gamma (1 / L):", step)
    print("Final iteration for x_k:", x_final)
    print("||grad(Q(x_0))||:", np.linalg.norm(gradient(x_0, P, c)))
    print("||grad(Q(x_k))||:", np.linalg.norm(gradient(x_final, P, c)))
    print("Total number of iterations:", iterations)
    print("||grad(Q(x_k))|| <= tolerance?", np.linalg.norm(gradient(x_final, P, c)) <=  tol) # Boolean
    print("Iterations < Max Iterations?", iterations < max_iter)     # Boolean

# Main script that generates the least squares problem, runs the desired 
# gradient descent algorithm and plots the convergence information into a figure.
def test_convergence():
    k_values = range(1, 11) # Matrix sizes to benchmark each method
    avg_iterations = []
    tol = 1e-4
    max_iter = 1e+5
    
    # IMPORTANT #
    # This loop will take a substantial amount of time to complete, and it might
    # appear that the program has stopped working. Patience is advised.
    for k in k_values:
        iterations_list = []
        for _ in range(3):      # Average over 3 runs
            # Different Method Parameters
            A, b, x_0 = generator(k)
            P, c, d = precompute(A, b)
            L = compute_L_constant(P)
            step = 1 / L
            p = 0.1
            alpha = 0.5
            beta = 0.5

            # Run desired method
            # Change line 364 manually for different methods #
            _, grad_norms, iterations = conjugate_gradient(A, b, x_0, tol, max_iter)
            iterations_list.append(iterations)

        avg_iterations.append(np.mean(iterations_list))
        print(f"Size 2^{k}: Average Iterations = {avg_iterations[-1]:.2f}")

    plotter(k_values, avg_iterations, method_name=r"Conjugate Gradient Method")

# Main script that calculates uptime (convergence successes) for each descent
# algorithm by taking into consideration the different maximum values of k
# (maximum matrix sizes for which the algorithms achieve convergence).
def uptime():
    tol = 1e-4
    max_iter = 1e+5
    # Hold functions and maximum matrix sizes in a dictionary
    methods = {
            "Constant Step": {"func": constant_gradient_descent},
            "Vanishing Step": {"func": vanishing_gradient_descent},
            "Exact Step": {"func": exact_gradient_descent},
            "Armijo Step": {"func": armijo_gradient_descent},
            "Conjugate Gradient": {"func": conjugate_gradient}
            }
    # Initialize empty dictionary that will store the results
    results = {}

    # Iterate over method names and details
    for method_name, details in methods.items():
        func = details["func"]
        k_values = range(1, 8)
        uptimes = []
        
        # Iterate over k, for different maximum kappas
        for k in k_values:
            success_count = 0       # Method convergent?
            total_runs = 3          # Total test runs
            
            for _ in range(total_runs):
                # Generate Problem parameters
                A, b, x = generator(k)

                # Zero initial state
                x0 = np.zeros(2**k)

                # Boolean variable
                converged = False

                # Test each method against its maximum k
                if method_name == "Constant Step":
                    L = compute_L_constant(precompute(A, b)[0])
                    step = 1 / L
                    converged = constant_gradient_descent(A, b, x0, step, tol, max_iter)[2] < max_iter
                elif method_name == "Vanishing Step":
                    L = compute_L_constant(precompute(A, b)[0])
                    step = 1 / L
                    p = 0.1
                    converged = vanishing_gradient_descent(A, b, x0, step, p, tol, max_iter)[2] < max_iter
                elif method_name == "Exact Step":
                    converged = exact_gradient_descent(A, b, x0, tol, max_iter)[2] < max_iter
                elif method_name == "Armijo Step":
                    alpha = 0.5
                    beta = 0.5
                    converged = armijo_gradient_descent(A, b, x0, alpha, beta, tol, max_iter)[2] < max_iter
                elif method_name == "Conjugate Gradient":
                    converged = conjugate_gradient(A, b, x0, tol, max_iter)[2] < max_iter
                if converged:
                    success_count += 1
            
            # Define uptime ratios for each method
            uptime_ratio = success_count / total_runs
            uptimes.append(uptime_ratio)
            # Print if the method succeeded or not
            print(f"{method_name} - k={k}: Uptime={uptime_ratio:.2f}")

        # Create dictionary entry
        results[method_name] = (list(k_values), uptimes)

    # Plot all uptime results for each method into the same figure
    plt.figure(figsize=(12, 8))
    for method_name, (k_values, uptimes) in results.items():
        plt.plot(k_values, uptimes, marker='o', linestyle='-', label=f"{method_name} (Max k={len(k_values)})")
    plt.xlabel("k (Matrix size 2^k)")
    plt.ylabel("Uptime (Convergence Success Rate)")
    plt.title("Uptime Comparison of Different Gradient Descent Methods")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

# Simple function that measures runtime of each gradient descent algorithm.
def measure_runtime(method_func, *args, **kwargs):
    start_time = time.time()
    result = method_func(*args, **kwargs)
    end_time = time.time()
    runtime = end_time - start_time
    return result, runtime
    
# Function that compates runtimes across different gradient descent methods
# for different matrix sizes.
def plot_runtime(methods, k_values, tol=1e-4, max_iter=1e5):
    results = {}  # Store runtimes for each method
    
    # Iterate over function names and details
    for method_name, details in methods.items():
        func = details['func']
        max_k = details['max_k']
        runtimes = []
        
        # Iterate over maximum matrix sizes for all methods
        for k in range(1, max_k + 1):
            # Initialize runtime variable
            total_runtime = 0
            
            # Do 3 test runs
            for _ in range(3):
                # Generate problem parameters
                A, b, x = generator(k)
                # Zero initial state
                x_0 = np.zeros(2**k)

                # Test over all methods
                if method_name == "Constant Step":
                    L = compute_L_constant(precompute(A, b)[0])
                    step = 1 / L
                    _, runtime = measure_runtime(constant_gradient_descent, A, b, x_0, step, tol, max_iter)
                
                elif method_name == "Vanishing Step":
                    L = compute_L_constant(precompute(A, b)[0])
                    step = 1 / L
                    p = 0.1
                    _, runtime = measure_runtime(vanishing_gradient_descent, A, b, x_0, step, p, tol, max_iter)
                
                elif method_name == "Exact Step":
                    _, runtime = measure_runtime(exact_gradient_descent, A, b, x_0, tol, max_iter)
                
                elif method_name == "Armijo Rule":
                    alpha = 0.5
                    beta = 0.5
                    _, runtime = measure_runtime(armijo_gradient_descent, A, b, x_0, alpha, beta, tol, max_iter)
                
                elif method_name == "Conjugate Gradient":
                    _, runtime = measure_runtime(conjugate_gradient, A, b, x_0, tol, max_iter)
                
                total_runtime += runtime
            
            # Average runtime per matrix size
            avg_runtime = total_runtime / 3
            # Append into results
            runtimes.append(avg_runtime)
            # Print average runtime for each iteration of k
            print(f"{method_name} - k={k}: Avg Runtime={avg_runtime:.4f} seconds")
        
        results[method_name] = (list(range(1, max_k + 1)), runtimes)
    
    # Plot Runtime Comparison
    plt.figure(figsize=(12, 8))
    for method_name, (k_values, runtimes) in results.items():
        plt.plot(
            k_values, runtimes, marker='o', linestyle='-', label=f"{method_name} (Max k={len(k_values)})"
        )
    
    plt.xlabel('k (Exponent in Matrix Size: 2^k)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Runtime Comparison of Gradient Descent Methods')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


# Compare runtime across multiple gradient descent methods up until the maximum
# matrix size that each method is convergent.
def main_runtime():
    methods = {
        'Constant Step': {'func': constant_gradient_descent, 'max_k': 7},
        'Vanishing Step': {'func': vanishing_gradient_descent, 'max_k': 6},
        'Exact Step': {'func': exact_gradient_descent, 'max_k': 7},
        'Armijo Rule': {'func': armijo_gradient_descent, 'max_k': 7},
        'Conjugate Gradient': {'func': conjugate_gradient, 'max_k': 10}
    }
    
    plot_runtime(methods, range(1, 11), tol=1e-4, max_iter=1e5)


# Simple menu function that calls the different main functions of this file.
def simple_cli():
    print("Simple Command Line Interface")
    print("=============================")
    print("1. System Testing (manipulate line 364 first).")
    print("2. Test Uptime (convergence successes vs. matrix size).")
    print("3. Test Runtime (compare runtimes of different descent algorithms).")
    prompt = input("Choose: ")
    if prompt == "1":
        test_convergence()
    elif prompt == "2":
        uptime()
    elif prompt == "3":
        main_runtime()
    else:
        print(f"{prompt} is not a valid choice")
        simple_cli()

simple_cli()
