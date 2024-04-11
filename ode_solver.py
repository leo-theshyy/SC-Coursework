import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.signal import find_peaks
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
def numerical_shooting(ode_func, t_span, initial_conditions, initial_guess, *args, tol=0.1, max_iter=1000, **kwargs):
    
    def objective_function(y, t_span):
        _, y_solution = ode_solver(ode_func, y, t_span, *args)
        return initial_guess - y_solution[-1]   
    
    
    # Perform shooting method
    for _ in range(max_iter):
        # Solve the ODE system with the current initial conditions
        solution = odeint(ode_func, initial_conditions, t_span, args, **kwargs)
        
        # Check the phase condition
        if np.allclose(solution[-1], initial_guess, atol=tol):
            # If phase condition is met, compute period and return
            period = compute_period(solution[:, 0], t_span)
            return initial_conditions, period
        
        # Use Newton's method to update the initial conditions
        initial_conditions -= (objective_function(initial_conditions, t_span) / np.gradient(ode_func(initial_conditions, t_span, *args)))
    
    raise ValueError("Numerical shooting did not converge within the maximum number of iterations.")

'''
compute the period
'''
def compute_period(solution, t_span):
    peaks, _ = find_peaks(solution)
    print(peaks)
    if len(peaks) < 2:
        # 如果找不到足够的峰值，返回一个默认值或者抛出异常
        return np.nan  
    period = (t_span[1]-t_span[0])*(peaks[1]-peaks[0])
    return period

'''
use scipy.solve_bvp
solve_bvp
'''
def bvp_residuals(ode_func, guess, t_span, x_end):
    sol = solve_ivp(lambda t, xy: ode_func(t, xy), t_span, [guess[0], guess[1]], t_eval=[t_span[1]])
    return sol.y[0][-1] - x_end

def solve_bvp(ode_func, t_span, x_end, guess):
    solution = minimize(bvp_residuals, guess, args=(ode_func, t_span, x_end), method='BFGS')
    return solution.x