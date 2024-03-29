import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

'''
solve_to
'''
# Perform a single Euler step
def euler_step(f, x0, t0, dt):
    return x0 + dt * f(x0, t0)

# Perform a single RK4 step
def rk4_step(f, x0, t0, dt):
    k1 = dt * f(x0, t0)
    k2 = dt * f(x0 + 0.5 * k1, t0 + 0.5 * dt)
    k3 = dt * f(x0 + 0.5 * k2, t0 + 0.5 * dt)
    k4 = dt * f(x0 + k3, t0 + dt)
    return x0 + (k1 + 2*k2 + 2*k3 + k4) / 6

# Solve the ODE x' = f from t0 to t1 using the Euler method or RK4
def solve_to(f, x0, t0, t1, dt_max, method):

    t_values = [t0]
    x_values = [x0]
    t = t0
    x = x0

    while t < t1:
        dt = min(dt_max, t1 - t)   # Prevent t from exceeding t1
        if method == 'euler':   # Choose method
            x = euler_step(f, x, t, dt)
        elif method == 'rk4':
            x = rk4_step(f, x, t, dt)
        t += dt
        t_values.append(t)
        x_values.append(x)

    return np.array(t_values), np.array(x_values)

'''
calculate_error
'''
# Analytical solution of the ODE
def analytical_solution(t):
    return np.exp(t)

# Calculate error based on method parameter
def calculate_error(f, x0, t0, t1, dt_values, method):
    errors = []

    for dt in dt_values:
        t_numerical, x_numerical = solve_to(f, x0, t0, t1, dt, method)
        x_analytical = analytical_solution(t_numerical)
        error = np.abs(x_numerical[-1] - x_analytical[-1])
        errors.append(error)

    return errors

'''
scipy.odeint
ode_solver
'''
def ode_solver(ode_func, initial_conditions, t_span, *args, **kwargs):
    solution = odeint(ode_func, initial_conditions, t_span, args, **kwargs)
    t = t_span if hasattr(t_span, '__len__') else [t_span[0], t_span[-1]]
    return t, solution

'''
numerical_shooting
'''
def numerical_shooting(ode_func, t_span, initial_guess, *args, tol=1e-6, max_iter=100, **kwargs):
    
    def objective_function(y, t_span):
        _, y_solution = ode_solver(ode_func, y, t_span, *args)
        return y_solution[-1] - initial_guess
    
    # Initial guess for the shooting method
    initial_conditions = initial_guess
    
    # Perform shooting method
    for _ in range(max_iter):
        # Solve the ODE system with the current initial conditions
        y_solution = odeint(ode_func, initial_conditions, t_span, args, **kwargs)
        
        # Check the phase condition
        if np.allclose(y_solution[-1], initial_guess, atol=tol):
            # If phase condition is met, compute period and return
            period = compute_period(y_solution)
            return initial_conditions, period
        
        # Use Newton's method to update the initial conditions
        initial_conditions -= (objective_function(initial_conditions, t_span) /
                                np.gradient(ode_func(initial_conditions, t_span, *args)))
    
    raise ValueError("Numerical shooting did not converge within the maximum number of iterations.")

'''
compute the period
'''
def compute_period(solution):
    # Assuming the solution contains at least two periods
    peak_indices = np.where(np.logical_and(solution[:-1] < solution[1:], solution[1:] < solution[:-1]))[0]
    period = np.mean(np.diff(peak_indices))
    return period