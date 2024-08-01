import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import root
from scipy.signal import find_peaks
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

def numerical_shooting_bru(ode_func, t_span, initial_conditions, initial_guess, *args, tol=1e-3, max_iter=1000):
    def objective_function(y, t_span):
        _, y_solution = ode_solver(ode_func, y, t_span, *args)
        return initial_guess - y_solution[-1]

    for _ in range(max_iter):
        solution = odeint(ode_func, initial_conditions, t_span, args=args)
        if np.allclose(solution[-1], initial_guess, atol=tol):
            period = compute_period(solution[:, 0], t_span)
            return initial_conditions, period
        initial_conditions -= (objective_function(initial_conditions, t_span) / np.gradient(ode_func(initial_conditions, t_span, *args)))
    raise ValueError("Numerical shooting did not converge within the maximum number of iterations.")

def ode_solver(ode_func, initial_conditions, t_span, *args):
    solution = odeint(ode_func, initial_conditions, t_span, args=args)
    return t_span, solution

def solve_limit_cycles(A, B_values):
    def equations(xy, A, B):
        x, y = xy
        return [A + x**2 * y - (B + 1) * x, B * x - x**2 * y]

    x_values, y_values = [], []
    for B in B_values:
        initial_condition = root(equations, [0, 0], args=(A, B)).x
        x_values.append(initial_condition[0])
        y_values.append(initial_condition[1])
    return B_values, x_values, y_values


def numerical_shooting(ode_func, t_span, initial_conditions, initial_guess, beta, tol=1e-3, max_iter=200):
    def objective_function(y, t_span,initial_guess):
        y_solution = odeint(ode_func, y, t_span, args=(beta,))
        return initial_guess - y_solution[-1]

    def compute_jacobian(f, y, t, beta, eps=1e-6):
        n = len(y)
        J = np.zeros((n, n))
        f0 = np.array(f(y, t, beta))
        for i in range(n):
            y_eps = y.copy()
            y_eps[i] += eps
            f_eps = np.array(f(y_eps, t, beta))
            J[:, i] = (f_eps - f0) / eps
        return J

    initial_conditions = initial_conditions.astype(float)  # Ensure initial_conditions is float
    for iteration in range(max_iter):
        solution = odeint(ode_func, initial_conditions, t_span, args=(beta,))
        residual = np.abs(objective_function(initial_conditions, t_span,initial_guess))
        print(f"Iteration {iteration}: residual = {residual}")
        if np.allclose(residual, 0, atol=tol):
            period = compute_period(solution[:, 0], t_span)
            return initial_conditions, period
        jacobian = compute_jacobian(ode_func, initial_conditions, t_span[0], beta)
        # Add regularization term to the Jacobian to prevent singular matrix error
        reg_term = np.eye(jacobian.shape[0]) * 1e-6
        try:
            step = np.linalg.solve(jacobian + reg_term, objective_function(initial_conditions, t_span,initial_guess))
            initial_conditions -= step
        except np.linalg.LinAlgError as e:
            print(f"Iteration {iteration}: LinAlgError - {e}")
            break
    raise ValueError("Numerical shooting did not converge within the maximum number of iterations.")

def compute_period(solution, t_span):
    peaks, _ = find_peaks(solution)
    if len(peaks) < 2:
        return np.nan
    period = (t_span[1] - t_span[0]) * (peaks[1] - peaks[0])
    return period



def poisson_rhs(x, sigma):
    """Calculate the right-hand side of the Poisson equation."""
    return -1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2 / (2 * sigma**2))

def finite_difference_poisson_root(sigma, grid_points):
    """Solve the Poisson equation using a finite difference method and root-finding."""
    D = 1
    x = np.linspace(-1, 1, grid_points)
    dx = x[1] - x[0]
    
    # Construct the coefficient matrix A using finite differences
    A = diags([1, -2, 1], [-1, 0, 1], shape=(grid_points-2, grid_points-2)).toarray()
    b = poisson_rhs(x[1:-1], sigma) * dx**2 / D
    
    # Initial guess for the solution
    u_guess = np.zeros(grid_points-2)
    
    # Use a root-finding method to solve the linear system
    solution = root(lambda u: A @ u + b, u_guess, method='hybr')
    
    # Construct the full solution including boundary conditions
    u = np.concatenate(([-1], solution.x, [-1]))
    
    return x, u

def finite_difference_poisson_sparse(sigma, grid_points):
    """Solve the Poisson equation using a finite difference method and a sparse matrix solver."""
    D = 1
    x = np.linspace(-1, 1, grid_points)
    dx = x[1] - x[0]
    
    # Construct the sparse matrix A using finite differences
    diagonals = [np.ones(grid_points-2), -2*np.ones(grid_points-2), np.ones(grid_points-2)]
    A = diags(diagonals, [-1, 0, 1], shape=(grid_points-2, grid_points-2))
    A = csc_matrix(A)  # Convert A to CSC format
    
    # Right-hand side of the equation
    b = poisson_rhs(x[1:-1], sigma) * dx**2 / D
    
    # Solve the linear system using a sparse matrix solver
    u_interior = spsolve(A, b)
    
    # Construct the full solution including boundary conditions
    u = np.concatenate(([-1], u_interior, [-1]))
    
    return x, u

def finite_difference_rcd(P, grid_points=81):
    """
    Solves the reaction-convection-diffusion equation using the finite difference method.

    Parameters:
    - P: The Peclet number which defines the strength of the convection.
    - grid_points: The number of points in the discretization grid.

    Returns:
    - x: The spatial grid.
    - u: The solution at each point on the grid.
    """
    x = np.linspace(0, 1, grid_points)
    dx = x[1] - x[0]
    
    # Construct the coefficient matrix A
    diagonals = [
        -2 * np.ones(grid_points),  # main diagonal
        np.ones(grid_points - 1),   # upper diagonal
        np.ones(grid_points - 1)    # lower diagonal
    ]
    A = diags(diagonals, [0, 1, -1]).toarray()
    A[0, 0] = 1  # Boundary condition u(0) = 0
    A[0, 1] = 0
    A[-1, -1] = 1  # Boundary condition u(1) = 1/2
    A[-1, -2] = 0
    A = csc_matrix(A) / dx**2
    
    # Construct the right-hand side vector b
    b = np.zeros(grid_points)
    b[-1] = 0.5
    b[1:-1] = P * (b[2:] - b[:-2]) / (2 * dx) - P

    # Solve the linear system A*u = b
    u = spsolve(A, b)
    
    return x, u