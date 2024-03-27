import numpy as np
import matplotlib.pyplot as plt
import time


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

# Analytical solution of the ODE
def analytical_solution(t):
    return np.exp(t)

# Calculate error based on method parameter
def calculate_error(method, dt_values):
    errors = []

    for dt in dt_values:
        t_numerical, x_numerical = solve_to(method, f, 1, t0, t1, dt)
        x_analytical = analytical_solution(t_numerical)
        error = np.abs(x_numerical[-1] - x_analytical[-1])
        errors.append(error)

    return errors


